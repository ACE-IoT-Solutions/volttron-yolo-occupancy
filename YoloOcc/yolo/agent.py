"""
Agent documentation goes here.
"""

__docformat__ = 'reStructuredText'

import logging
import sys
import os
import json
import base64
from volttron.platform.agent import utils
from volttron.platform.vip.agent import Agent, Core, RPC
from PIL import Image
import grequests
import requests
import gevent
from requests.auth import HTTPDigestAuth
from ultralytics import YOLO
from io import BytesIO
from datetime import datetime
from volttron.platform.messaging import headers as header_mod
from volttron.platform.agent import utils

_log = logging.getLogger(__name__)
utils.setup_logging()
__version__ = "0.1"


def yolo(config_path, **kwargs):
    """
    Parses the Agent configuration and returns an instance of
    the agent created using that configuration.

    :param config_path: Path to a configuration file.
    :type config_path: str
    :returns: Yolo
    :rtype: Yolo
    """
    try:
        config = utils.load_config(config_path)
    except Exception:
        config = {}

    if not config:
        _log.info("Using Agent defaults for starting configuration.")

    camera_list = config.get('camera_list', [])
    scan_interval = int(config.get('scan_interval', "300"))
    site = config.get('site', 'sitetest')
    client = config.get('client', 'clienttest')
    filter_items = config.get('filter_items', [])
    conf_threshold = config.get('conf_threshold', 0)
    ai_model_path = config.get('ai_model_path',"yolov8n.pt")
    return Yolo(camera_list, scan_interval, site, client, filter_items, conf_threshold, ai_model_path, **kwargs)


class Yolo(Agent):
    """
    Document agent constructor here.
    """

    def __init__(self, camera_list=[], scan_interval=300, site = 'test_site', client = 'test_client', filter_items = [], conf_threshold=0, ai_model_path = "yolov8n.pt", **kwargs):
        super(Yolo, self).__init__(enable_web=True, **kwargs)
        _log.debug("vip_identity: " + self.core.identity)

        self.camera_list = camera_list
        self.scan_interval = scan_interval
        self.client = client
        self.site = site
        self.camera_analysis = None
        self.filter_items = filter_items
        self.conf_threshold = conf_threshold
        self.ai_model_path = ai_model_path

        self.default_config = {"camera_list": camera_list,
                               "scan_interval": scan_interval}
        
        self.model = YOLO(self.ai_model_path)


        # Set a default configuration to ensure that self.configure is called immediately to setup
        # the agent.
        self.vip.config.set_default("config", self.default_config)
        # Hook self.configure up to changes to the configuration file "config".
        self.vip.config.subscribe(self.configure, actions=["NEW", "UPDATE"], pattern="config")

    def configure(self, config_name, action, contents):
        """
        Called after the Agent has connected to the message bus. If a configuration exists at startup
        this will be called before onstart.

        Is called every time the configuration in the store changes.
        """
        config = self.default_config.copy()
        config.update(contents)


        # _log.debug("Configuring Agent")
        # _log.debug(contents)
        try:
            # if config_name == "config":
            #     for entry in contents:
            #         _log.debug(f"setting {entry}")
            self.camera_list = contents.get("camera_list")
            self.scan_interval = contents.get("scan_interval", 30)
            self.site = contents.get("site")
            self.client = contents.get("client")
            self.filter_items = contents.get("filter_items", [])
            self.conf_threshold = contents.get("conf_threshold", 0)
            self.ai_model_path = contents.get("ai_model_path", "yolov8n.pt")
            self.model = YOLO(self.ai_model_path)
            if self.camera_analysis is not None:
                self.camera_analysis.kill()
            self.camera_analysis = self.core.periodic(self.scan_interval, self.send_camera_results)
        except ValueError as e:
            _log.error("ERROR PROCESSING CONFIGURATION: {}".format(e))
            return

    def _create_subscriptions(self, topic):
        """
        Unsubscribe from all pub/sub topics and create a subscription to a topic in the configuration which triggers
        the _handle_publish callback
        """
        self.vip.pubsub.unsubscribe("pubsub", None, None)

        self.vip.pubsub.subscribe(peer='pubsub',
                                  prefix=topic,
                                  callback=self._handle_publish)

    def _handle_publish(self, peer, sender, bus, topic, headers, message):
        """
        Callback triggered by the subscription setup using the topic from the agent's config file
        """
        pass

    def _grequests_exception_handler(self, request, exception):
        """
        Log exceptions from grequests
        """
        _log.error(f"grequests error: {exception} with {request}")

    def analyze_images(self, image:Image, camera_name):
        def center_point(x1, y1, x2, y2):
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            return (x_center, y_center)

        def check_dict(d, key):
            if key in d:
                d[key] += 1
            else:
                d[key] = 1
            return d

        def store_image_quadrant(x, y, W, H, quadrant_dict, key):
            if x < W/2 and y < H/2:
                return check_dict(quadrant_dict, "top-left-quadrant/" + key)
            elif x >= W/2 and y < H/2:
                return check_dict(quadrant_dict, "top-right-quadrant/" + key)
            elif x < W/2 and y >= H/2:
                return check_dict(quadrant_dict, "bottom-left-quadrant/" + key)
            elif x >= W/2 and y >= H/2:
                return check_dict(quadrant_dict, "bottom-right-quadrant/" + key)
    
        results = self.model.predict(image, save=True, project=f'images/yoloimg/{self.site}', name=camera_name, exist_ok = True)[0]
        identified_items = {}
        # _log.debug(results)
        if results.boxes:
            boxes = results.boxes.cpu().numpy()
            # _log.debug('BOXES FOUND///////////////////////')

            for box in boxes:
                box_coordinates = box.xyxy[0].astype(int)
                box_identified = results.names[int(box.cls[0])]
                # _log.debug(box_coordinates)

                store_box_identified = False
                if (not self.filter_items or box_identified in self.filter_items) and box.conf[0] > self.conf_threshold:
                    store_box_identified = True

                if store_box_identified:
                    check_dict(identified_items, 'total/' + box_identified)
                    center_x, center_y = center_point(box_coordinates[0], box_coordinates[1], box_coordinates[2], box_coordinates[3])
                    store_image_quadrant(center_x, center_y, results.orig_shape[1], results.orig_shape[0], identified_items, box_identified)
        return identified_items

    def send_camera_results(self):
        for camera in self.camera_list:
            auth = HTTPDigestAuth(camera.get('username'), camera.get('password'))
            _log.debug('username: ' + camera.get('username'))
            # response = requests.get(camera.get('url'), auth=auth, verify=False)
            req = grequests.get(camera.get('url'), auth=auth, verify=False)
            (response,) = grequests.map(
                (req,), exception_handler=self._grequests_exception_handler
            )
            if response and response.status_code == 200:
                image_bytes = BytesIO(response.content)
                image = Image.open(image_bytes)
                filename = f"current_image.jpg"
                image.save(filename, format='JPEG')
                analysis_result = self.analyze_images(filename, camera.get('name'))
                analysis_result['online'] = 1
                # _log.debug("Response received")
            else:
                # if response:
                #     _log.debug(response.status_code)
                #     _log.debug(response.text)
                analysis_result = {'online': 0}
            now = utils.format_timestamp( datetime.utcnow())
            header = {
                header_mod.DATE: now,
                header_mod.TIMESTAMP: now
            }
            # _log.debug(analysis_result)
            self.vip.pubsub.publish(
                'pubsub', 
                f"devices/{self.client}/{self.site}/cameras/{camera.get('name')}/all",
                headers=header,
                message=[analysis_result]
            )
        return


    def jsonrpc(self, env, data):
        """
        Returns camera information for web interface
        """
        data = []
        for camera in self.camera_list:
            camera_url = f"/yoloimg/{self.site}/{camera['name']}/current_image.jpg"
            data.append({
                "name": camera['name'],
                "src": camera_url
            })
        return {'data' : data}



    @Core.receiver("onstart")
    def onstart(self, sender, **kwargs):
        """
        This is method is called once the Agent has successfully connected to the platform.
        This is a good place to setup subscriptions if they are not dynamic or
        do any other startup activities that require a connection to the message bus.
        Called after any configurations methods that are called at startup.

        Usually not needed if using the configuration store.
        """
        # Example publish to pubsub
        # self.vip.pubsub.publish('pubsub', "devices/camera/topic", message="HI!")
        _log.debug("in onstart")
        _log.debug(f"{self.scan_interval=}")

        # Sets WEB_ROOT to be the path to the webroot directory
        # in the agent-data directory of the installed agent..
        WEB_ROOT = os.path.abspath(os.path.abspath(os.path.join(os.path.dirname(__file__), 'webroot/')))
        # Serves the static content from 'webroot' directory

        self.vip.web.register_path(r'^/yolo/.*', WEB_ROOT)
        self.vip.web.register_endpoint(r'/yolorpc/jsonrpc', self.jsonrpc)

        camera_image_roots = os.path.abspath(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', f'images/')))
        os.makedirs(camera_image_roots, exist_ok = True)
        _log.debug(camera_image_roots)
        self.vip.web.register_path(r'^/yoloimg/.*', camera_image_roots)
        # Example RPC call
        # self.vip.rpc.call("some_agent", "some_method", arg1, arg2)
        # pass

    @Core.receiver("onstop")
    def onstop(self, sender, **kwargs):
        """
        This method is called when the Agent is about to shutdown, but before it disconnects from
        the message bus.
        """
        pass

    @RPC.export
    def rpc_method(self, arg1, arg2, kwarg1=None, kwarg2=None):
        """
        RPC method

        May be called from another agent via self.core.rpc.call
        """
        pass
        # return self.setting1 + arg1 - arg2


def main():
    """Main method called to start the agent."""
    utils.vip_main(yolo, 
                   version=__version__)


if __name__ == '__main__':
    # Entry point for script
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass

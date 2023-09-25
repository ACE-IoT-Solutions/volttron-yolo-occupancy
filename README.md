# YoloOcc VOLTTRON™️ Agent

![ACE IoT Solution logo](https://github.com/ACE-IoT-Solutions/volttron-yolo-occupancy/blob/main/ace_iot_solutions.png?raw=true)

This project provides an Eclipse VOLTTRON™️ Agent for determining occupancy count of a space using the YoloV8 Model and nVidia CUDA acceleration.
The model will also run on CPU resources, but will be considerably slower. The application was tested on a Jetson Nano 8GB Developer Kit, where it delivered frame rates of 10-15 FPS, or 10-15 different camera feeds at 1 FPS.
The camera interface was designed for Hikvision DS-2CD2185FWD-I 8MP cameras, but should work with any camera that supports a simple JPEG snapshot URL.
The interface was also designed to be easily extensible for other authentication methods.

![Sample image of cube farm with raw model output overlaid](https://github.com/ACE-IoT-Solutions/volttron-yolo-occupancy/blob/main/sample_cube_farm.png?raw=true)


This project was created for [Slipstream Inc](https://slipstreaminc.org), as a collaboration with the [Minnesota Center for Energy and the Environment (MCEE)](https://www.mncee.org/wi-fi-location-based-services-optimize-energy-efficiency) and is supported in part by a grant from the U.S. Department of Energy’s Office of Energy
Efficiency and Renewable Energy under the Award Number
EE0008684.

## Installation
To use CUDA, you will need to follow the instructions for setting up the required libraries for your environment. You will also need to make sure the user that volttron is running under has access to the CUDA libraries and is a member of the `video` group.

The agent can be installed using the VOLTTRON™️ Control Panel, or by running the following command from the VOLTTRON™️ root directory:

```python scripts/install-agent.py -s YoloOcc -i yoloocc -t yoloocc -f```

Then adding the configuration file to the VOLTTRON™️ configuration store:

```volttron-ctl config store yoloocc config <path to config file>```

By default, the agent will download a pre-trained YoloV8 model from Ultralytics. If you wish to use a custom model, you can provide the path to the model in the configuration file, as shown below. Best results were found with the model file included in this repo for 1920x1080 images.

## Configuration
A sample config is provided in the repository. The config file is a JSON file with the following fields:
* `camera_list`: A list of camera definitions with the following fields:
    * `url`: The URL to the camera snapshot. The URL should return a JPEG image.
    * `username`: The username for the camera. If the camera does not require authentication, this field can be omitted.
    * `password`: The password for the camera. If the camera does not require authentication, this field can be omitted.
    * `name`: The name of the camera. This will be used as the topic for the occupancy count.
    * `description`: Description of the camera. This will be used as the description for the occupancy count meta data.
    * `auth_method`: The authentication method to use for the camera. Currently only `basic` and `digest` are supported, default is `digest`.
* `ai_model_path`: The path to the YoloV8 model. The model can be downloaded from Ultralytics or you can provide a custom model
* `conf_threshold`: The confidence threshold for the model. This is the minimum confidence for a detection to be considered valid.
* `scan_interval`: The interval at which the agent will poll the cameras for snapshots.
* `client`: The client name to use for the agent, this will be used to build the topic structure for the data published to the message bus.
* `site`: The site name to use for the agent, this will be used to build the topic structure for the data published to the message bus.

Example below:
```
{
    "camera_list": [
        {
            "name": "andrew_office",
            "url": "https://192.168.9.132/ISAPI/Streaming/channels/1/picture?videoResolutionWidth=1920&videoResolutionHeight=1080",
            "username": "admin",
            "password": "#8jrh%A#eo&CeuV3",
            "description": "example camera of andrew's office"
        }
    ],
    "scan_interval": 20,
    "site": "site123",
    "client": "client123",
    "conf_threshold": 0.3,
    "ai_model_path": "/var/lib/volttron/YoloOcc/yolo/models/yolov8m.pt"
}
```
# pelvis-fluoro-software

This repository contains my work over the 2024 spring semester of CIS II. This includes a backend to obtain X-rays from a video capture device, process X-rays through a provided model, and send analysis over a websocket to a locally-hosted frontend that displays all visualizations and key measurements.

Software diagram:
![image1](https://github.com/janyabudaraju/pelvis-fluoro-software/assets/52179416/1dad92b2-09a6-44a5-aec5-25adb2a87274)


## dependencies
- **Node.js**: [download & install Node.js](https://nodejs.org/)
- run `pip3 install -r requirements.txt` to obtain all python library dependencies.
- run `npm install` in `UI` directory to obtain all js library dependencies.
- navigate to `dependencies/perphix` and run `pip3 install .`
- ensure that you have a `.pth` model weights file in the `dependencies` directory.

## running
- ensure that a video capture device is connected on port specified in `main.py`.
- run the command `npm run dev` in `UI` directory to start up the frontend.
- run `main.py` to start up the backend.

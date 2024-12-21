# Data Labelling
## Overview
This directory provides the necessary tools for part selection data labelling (See the [Additional Experiments](https://asap.csail.mit.edu/#:~:text=Additional%20Experiments) section for a more detailed overview).

## Preparing Assembly Data for Labelling
### Assembly Data Directory
To prepare an assembly dataset to be labelled, make sure the assembly meshes are stored in the correct format. In the assembly directory, store each part in the assembly as a separate `.obj` file, along with a `config.json` file which stores the state and center of mass of each part in the assembly. See the [beam assembly directory](assets/beam_assembly/original/) as an appropriate example.

### Converting to GLB format 
Next, we use the `generate_unlabelled_dataset.py` to convert the assembly into GLB format such that it can be loaded by Three.JS in the data labelling Webpage.

Define the `assembly_name`, and also `assembly_obj_dir` as the path to the assembly directory created in the previous step, and run `python generate_unlabelled_dataset.py`.

This script will create the `viz_data.json` file inside your assembly directory which takes the following structure:
```json
{
    "assembly-id": assembly name defined above,
    "parts": list with names of each part in assembly,
    "model-data": full assembly mesh data in GLB format,
}
```

## Data Labelling Webpage
The Data Labelling Webpage provides the functionality to observe the assembly and label the appropriate parts for disassembly.

To view the data labelling webpage, grab the URL of the `viz_data.json` file created in the previous step and use it to define the `OBJ_URL` variable in `picker.html` (line 271).

More instructions on how to interact with the webpage can be found under the `Instructions` and `How to use 3D Model Viewer` tabs in the webpage.

Currently selecting the part and submitting only stores the part name in the `removed-part` input box. The webpage can be modified to store the results in a local json file or call an API to record the results. It can also be used with mTurk workflow.

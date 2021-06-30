# computer-vision-2021

## final_project: Boat detection and sea segmentation

### Requirements

Install Xcode tools + brew

```zsh
brew tap nlohmann/json
brew install cmake eigen nlohmann-json opencv miniforge
conda init cv
conda activate cv
pip install tensorflow-macos tensorflow-metal pillow
```

#### Datasets

Boat detection: [OpenImages v6](https://storage.googleapis.com/openimages/web/index.html)
Sea segmentation: [ADE20K Outdoors](https://www.kaggle.com/residentmario/ade20k-outdoors)

### Build

```zsh
cd final_project
mkdir build && cd build
cmake .. && make
```

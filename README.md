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

The work for the sea segmentation part has been distibuted across the second half of June and first half of July. The actual time spent on it is roughly 20 hours.

For performance reasons, the test images have been reduced in size to .... px of width.

To train the CNN classifier, The dataset ADE20K Outdoors is used (https://www.kaggle.com/residentmario/ade20k-outdoors), that consists of 5000 images with segmentation labels. For the training the classes "sea" and "water" are considered positive.

Unfortunately the training performed poorly, with accuracy just below 0.8. The average pixel accuracy for the Kaggle ships dataset is .... while the average pixel accuracy for the Venice dataset is .... .

\begin{center}
\begin{tabular}{ c c }
Kaggle dataset & Pixel accuracy
\hline
aida-ship-driving-cruise-ship-sea-144796 & cell2 \\
blue-boat-freedom-horizon-ocean-2878 & cell5 \\
boat-ferry-departure-crossing-sea-2733061 & cell8 \\
boat-haze-ship-alone-marine-water-1819696 & cell8 \\
caribbean-sea-travel-vacations-2712423 & cell8 \\
ferries-shipping-transport-cross-53122 & cell8 \\
oil-tankers-supertankers-oil-tankers-336718 & cell8 \\
ship-tanker-cargo-sea-2573453 & cell8 \\
water-inflatable-boat-sea-boa-yet-199811 & cell8
\end{tabular}
\end{center}

\begin{center}
\begin{tabular}{ c c }
Venice dataset & Pixel accuracy
\hline
00 & cell8 \\
01 & cell8 \\
02 & cell8 \\
03 & cell8 \\
04 & cell8 \\
05 & cell8 \\
06 & cell8 \\
07 & cell8 \\
08 & cell8 \\
09 & cell8 \\
10 & cell8 \\
11 & cell8
\end{tabular}
\end{center}


The program has been tested on an Apple Mac Mini with Apple Silicon SoC. Sample terminal interaction:

\begin{verbatim}
$ cd final_project/build
$ ./final_project sea_train ../data ../out
Find which images have a "sea" or "water" class...
Sampling images with sea... 1/1054
[...]
Sampling images with sea... 1054/1054
Sampling images without sea... 1/3946
[...]
Sampling images without sea... 3946/3946
Init Plugin
Init Graph Optimizer
Init Kernel

Compiling model...
Metal device set to: Apple M1
[...]

Indexing dataset...
[...]
302/302 [==============================] - 43s 139ms/step - loss: 0.5700 - accuracy: 0.7948
15/15 [==============================] - 2s 126ms/step - loss: 0.5477 - accuracy: 0.7872

Loss: 0.5476818084716797
Accuracy: 0.7871875238418579
[...]
  function_optimizer: function_optimizer did nothing. time = 0.002ms.
  function_optimizer: function_optimizer did nothing. time = 0ms.
$ ./final_project sea_segment ../out/model.pb ../data/Kaggle_ships/aida-ship-driving-cruise-ship-sea-144796.jpg ../out ../data/Kaggle_ships_mask/aida-ship-driving-cruise-ship-sea-144796.jpg
Progress: 0%
Progress: 0%
[...]
Progress: 99%
Progress: 99%
Pixel accuracy: 0.561461

Done
\end{verbatim}
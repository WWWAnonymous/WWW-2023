# A Flexible Measurement-based Modeling Generator

### About

In order to facilitate the interested reader to better apply our measurement findings, we have implemented the complete modeling generator based on Python and provided the necessary quantitative data to support it. In addition, this project uses GDP as well as geographic data from some regions of the USA as an example to help readers understand the implementation details. Hope this project can provide a real experience for the research and application of edge computing.

> Large-Scale Measurement of a Commercial Edge Computing Platform: Consolidation from Servers, Services and Requests (Anonymized for double-blind review)

Note: The GDP and geographic data used in this project are obtained from Wikipedia.

### Prerequisites

The code runs on Python 3. To install the dependencies, please execute:

```
pip3 install -r requirements.txt
```

### Project

- quantified_data - Includes the measurement data to support the project
- input_data - Includes GDP/population and geographic data for the target area as input, and currently is some regions of the USA
- output_data - includes the output result data obtained by running the project

### Getting Started

* Input data: You can use the population/GDP and geographic data of the target area as input. At the same time, we also provide the data sample in ./input_data.
* Parameter setting: Set up in main.py according to your own needs.
* Run:  ` python3 main.py`
* Result: Read and analyze the file in "\out".

### Version
* 0.1 beta

### Citation

If this paper can benefit your scientific publications, please kindly cite it.
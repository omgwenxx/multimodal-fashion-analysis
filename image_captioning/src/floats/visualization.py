"""
Based on the code from https://github.com/Tommel71/Master-Thesis-Data-Science/blob/main/MasterThesis/src/Datamodels/Visualisation.py
"""

from matplotlib import pyplot as plt
import seaborn as sns
import os
import pickle
import configparser
import pathlib
import pandas as pd


class Visualization:
    """
    Superclass for visualizations to be rendered and saved to latex folder for later use in master thesis.

    Attributes:
        name (str): The name of the visualization.
        data_name (str): The name of the data.
        chapter (str): The chapter of the master thesis.
        scaling_factor (float, optional): A scaling factor for the visualization. Defaults to 1.
        prefix (str, optional): prefix for paths used in this class e.g. config file
        outputfolder (str): The output folder path for saving visualizations.
        inputfile (str): The input folder path for data file used for visualization.
    """

    def __init__(self, name, data_name, chapter="chapter", scaling_factor=1, prefix=".."):
        """
        Initializes an instance of Visualization.

        Parameters:
            name (str): The name of the visualization.
            data_name (str): The name of the data.
            chapter (str): The chapter of the master thesis.
            scaling_factor (float, optional): A scaling factor for the visualization. Defaults to 1.
        """
        self.name = name
        self.data_name = data_name
        self.scaling_factor = scaling_factor
        self.chapter = chapter
        config = configparser.ConfigParser()
        config.read(f"{prefix}/config.ini")
        print(config.sections())
        self.plotting_config = config["Plotting"]
        self.outputfolder = os.path.join(f"{prefix}/../../latex/img/{self.chapter}")
        self.inputfile = None

    def set_outputfolder(self, new_path):
        self.outputfolder = new_path

    def set_inputfile(self, path):
        self.inputfile = path

    def set_settings(self):
        """
        Sets matplotlib parameters with values given by config file
        """
        sns.set_style("whitegrid")
        plt.rcParams.update(
            {
                "xtick.bottom": True,
                "ytick.left": True,
                "axes.labelsize": self.scaling_factor * self.plotting_config["label_size"],
                "font.size": self.scaling_factor * self.plotting_config["font_size"],
                "legend.fontsize": self.scaling_factor * self.plotting_config["legend_size"],
                "xtick.labelsize": self.scaling_factor * self.plotting_config["tick_size"],
                "ytick.labelsize": self.scaling_factor * self.plotting_config["tick_size"],
                "figure.titlesize": self.scaling_factor * self.plotting_config["title_size"],
                "axes.titlesize": self.scaling_factor * self.plotting_config["title_size"],
                "figure.dpi": self.plotting_config["figure_dpi"],
            }
        )

    def create_visualization(self):
        """
        To be coded in subclasses, should contain visualization code
        """
        pass

    def save_visualization(self):
        """
        Saves plot as pdf file to output folder property with the name specified via name property
        """
        os.makedirs(self.outputfolder, exist_ok=True)
        plt.savefig(
            f"{self.outputfolder}/{self.data_name}-{self.name}.pdf",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()

    def load_data(self):
        """
        Loads in data specified via inputfile (supports .pkl and .csv).
        """
        data = None  # Initialize data with a default value
        if self.inputfile is None:
            raise Exception("Data File not specified")

        try:
            file_path = pathlib.Path(self.inputfile)
            file_ext = file_path.suffix
            file_name = file_path.name
            print(f"Loading in file {file_name} (a {file_ext} file)")
            if file_ext == ".pkl":
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
            elif file_ext == ".csv":  # Changed to elif to ensure only one condition is true
                data = pd.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported file extension: {file_ext}")
        except FileNotFoundError:
            raise FileNotFoundError(f"File {file_path} not found")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

        return data

    def render_visualization(self):
        """
        Runs full of settings visualization settings, rendering plot and saving it to the
        outputfolder specified by the outputfolder property.
        """
        print("Set Settings")
        self.set_settings()
        print("Create Vis")
        self.create_visualization()
        print("Saved Vis")
        self.save_visualization()

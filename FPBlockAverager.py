#!/usr/bin/python
# -*- coding: utf-8 -*-
#   fpblockaverager.FPBlockAverager.py
#
#   Copyright (C) 2012-2016 Karl T Debiec
#   All rights reserved.
#
#   This software may be modified and distributed under the terms of the
#   BSD license. See the LICENSE file for details.
"""
Command-line tool to estimate standard error using the
block-averaging method of Flyvbjerg and Petersen

Flyvbjerg, H., and Petersen, H. G. Error estimates on averages of
correlated data. Journal of Chemical Physics. 1989. 91 (1). 461-466.
"""
################################### MODULES ###################################
from __future__ import absolute_import,division,print_function,unicode_literals
if __name__ == "__main__":
    __package__ = str("fpblockaverager")
    import fpblockaverager
import numpy as np
import pandas as pd
################################### CLASSES ###################################
class FPBlockAverager(object):
    """
    Class to manage estimation of standard error using the
    block-averaging method of Flyvbjerg and Petersen

    Flyvbjerg, H., and Petersen, H. G. Error estimates on averages of
    correlated data. Journal of Chemical Physics. 1989. 91 (1). 461-466.

    .. todo:
      - Support for multiprocessing would be nice
      - Possibly support wwmgr? Could be nice exercise
      - Test Python 3
      - Support myplotspec for formatting?
      - Support omitting blockings outside min_n_blocks and max cut from
        fit, but still calculating
      - Use a decorator to pull datasets, blockings, and fits from self?
      - Fit to linear portion after initial fits
      - Estimate sample size and correlation time
    """
    from . import arg_or_attr

    def __call__(self, **kwargs):
        """
        Carries out standard error calculation.

        Arguments:
          kwargs (dict): Additional keyword arguments

        .. todo:
          - loop over infile arguments
        """
        verbose = kwargs.get("verbose", 1)
        debug = kwargs.get("debug", 1)
        for infile in kwargs.pop("infile"):
            dataset   = self.load_datasets(infile=infile, **kwargs)
            blockings = self.select_blockings(dataset=dataset,**kwargs)
            if debug >= 1:
                print(blockings)
            blockings = self.calculate_blockings(dataset=dataset,
              blockings=blockings, **kwargs)
            blockings, parameters = self.fit_curves(dataset=dataset,
              blockings=blockings, **kwargs)
            self.plot(blockings=blockings, parameters=parameters, **kwargs)

    def __init__(self, dataframe=None, **kwargs):
        """
        """
        if dataframe is not None:
            print(dataframe)
            blockings = self.select_blockings(dataframe)
            print(blockings)

    def load_datasets(self, infile, verbose=1, **kwargs):
        """
        Load datasets from text, numpy, or hdf formats.

        .. todo:
          - Check for file presence and raise nice errors
          - Support text, npy, hdf5
        """
        from os.path import expandvars, isfile

        path = expandvars(infile[0])
        if not isfile(path):
            raise Exception("infile not found")
        if path.endswith(".h5") or path.endswith(".hdf5"):
            if len(infile) < 2:
                raise Exception("need address within h5")
            address = infile[1]
            dataset = pd.read_hdf(path, address)
            if len(infile) > 2:
                fields = infile[2:]
                dataset = dataset[fields]
                if verbose >= 1:
                    print("Dataset '{0}[{1}][{2}]' loaded".format(path,
                      address, ", ".join(fields)))
            else:
                if verbose >= 1:
                    print("Dataset '{0}[{1}]' loaded".format(path, address))
            if verbose >= 2:
                print(dataset)
        else:
            raise Exception("only hdf5 input currently supported")

        return dataset

#    @arg_or_attr("dataset")
    def select_blockings(self, dataframe, min_n_blocks=1, max_cut=0.1,
        all_factors=False, **kwargs):
        """
        Selects lengths of block-transformed dataframe

        Arguments:
          kwargs (dict): Additional keyword arguments
        """

        full_length = dataframe.shape[0]

        # Determine number of blocks, block lengths, total lengths used,
        #   and number of transforms
        if all_factors:
            block_lengths = np.array(sorted(set(
              range(1, 2 ** int(np.floor(np.log2(full_length)))))), np.int)
            n_blocks = np.array(sorted(set(
              np.array([full_length/n for n in block_lengths],
              np.int))))[::-1][:-1]
            n_blocks = n_blocks[n_blocks >= min_n_blocks]
        else:
            block_lengths = np.array([2 ** i for i in
              range(int(np.floor(np.log2(full_length))))], np.int)
            n_blocks = np.array([full_length/n for n in block_lengths], np.int)
            n_blocks = n_blocks[n_blocks >= min_n_blocks]
        block_lengths = np.array(full_length / n_blocks, np.int)
        used_lengths = n_blocks * block_lengths
        n_transforms = np.log2(block_lengths)

        # Cut blockings fot which more than max_cut proprotion of
        #   dataframe must be omitted
        max_cut_indexes = np.where(used_lengths / full_length >= 1-max_cut)[0]
        n_blocks      = n_blocks[max_cut_indexes]
        block_lengths = block_lengths[max_cut_indexes]
        used_lengths  = used_lengths[max_cut_indexes]
        n_transforms  = n_transforms[max_cut_indexes]

        # Organize and return
        blockings = pd.DataFrame(
          np.column_stack((n_transforms,n_blocks,block_lengths,used_lengths)),
          columns=["n_transforms", "n_blocks", "block_length", "used_length"])

        return blockings

    @arg_or_attr("dataset", "blockings")
    def calculate_blockings(self, dataset, blockings, **kwargs):
        """
        Calculates standard error for each block transform.

        .. note:: 
            The standard deviation of each standard error
            (stderr_stddev) is only valid for points whose standard
            error has leveled off (i.e. can be assumed Gaussian).

        Arguments:
          kwargs (dict): Additional keyword arguments
        """

        # Construct destination for results
        columns = [[c+"_mean", c+"_se", c+"_se_sd"] for c in dataset.columns]
        columns = [item for sublist in columns for item in sublist]
        analysis = pd.DataFrame(
          np.zeros((blockings.shape[0], dataset.shape[1]*3)), columns=columns)

        # Calculate mean, stderr, and stddev of stderr for each blocking
        for i, row in blockings.iterrows():
            transformed = self.transform(n_blocks=row["n_blocks"],
              block_length=row["block_length"], dataset=dataset, **kwargs)
            mean          = np.mean(transformed.values, axis = 0)
            stddev        = np.std(transformed.values,  axis = 0)
            stderr        = stddev / np.sqrt(row["n_blocks"] - 1)
            stderr_stddev = stderr / np.sqrt(2 * (row["n_blocks"] - 1))
            analysis.values[i,0::3] = mean
            analysis.values[i,1::3] = stderr
            analysis.values[i,2::3] = stderr_stddev

        # Organize and return
        return blockings.join(analysis)

    @arg_or_attr("dataset")
    def transform(self, n_blocks, block_length, dataset=None, **kwargs):
        """
        Prepares a block-transformed dataset.

        Arguments:
          block_length (int): Length of each block in transformed
            dataset
          n_blocks (int): Number of blocks in transformed dataset 
          used_length (int): Number of frames in transformed dataset
          kwargs (dict): Additional keyword arguments

        .. todo:
          - Is there an appropriate way to do this using pandas?
        """

        reshaped = np.reshape(dataset.values[:n_blocks*block_length],
          (n_blocks, block_length, dataset.shape[1]))
        transformed = pd.DataFrame(np.mean(reshaped, axis=1),
          columns=dataset.columns)
        return transformed

    @arg_or_attr("dataset", "blockings")
    def fit_curves(self, dataset, blockings, fit_exp=True,
        fit_sig=True, verbose=1, debug=0, **kwargs):
        """
        Fits exponential and sigmoid curves to block-transformed data.

        Arguments:
          kwargs (dict): Additional keyword arguments

        .. todo:
          - if exp a or sig b is negative, set to NaN and print a
            warning
        """
        import warnings
        from scipy.optimize import curve_fit

        def exponential(x, a, b, c):
            """
                         (c * x)
            y = a + b * e

            Arguments:
              x (float): x
              a (float): Final y value; y(+∞) = a
              b (float): Scale
              c (float): Power

            Returns:
              y (float): y(x)
            """
            return a + b * np.exp(c * x)

        def sigmoid(x, a, b, c, d):
            """
                     a - b
            y = --------------- + b
                           d
                1 + (x / c)

            Arguments:
              x (float): x
              a (float): Initial y value; y(-∞) = a
              b (float): Final y value; y(+∞) = b
              c (float): Center of sigmoid; y(c) = (a + b) / 2
              d (float): Power

            Returns:
              y (float): y(x)
            """
            return b + ((a - b) / (1 + (x / c) ** d))

        # Construct destinations for results
        fields = dataset.columns.tolist()
        columns = ["n_transforms", "n_blocks", "block_length", "used_length"]

        if fit_exp:
            exp_fit = pd.DataFrame(
              np.zeros((blockings.shape[0], dataset.shape[1]))*np.nan,
              columns=[f+"_exp_fit" for f in fields])
            exp_par = pd.DataFrame(np.zeros((3, len(fields)))*np.nan,
              index=["a (se)", "b", "c"], columns=fields)
        if fit_sig:
            sig_fit = pd.DataFrame(
              np.zeros((blockings.shape[0], dataset.shape[1]))*np.nan,
              columns=[f+"_sig_fit" for f in fields])
            sig_par = pd.DataFrame( np.zeros((4, len(fields)))*np.nan,
              index=["a", "b (se)", "c", "d"], columns=fields)

        # Calculate and store fit and parameters
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, field in enumerate(fields):
                columns.extend([field+"_mean", field+"_se", field+"_se_sd"])
                if fit_exp:
                    columns.append(field+"_exp_fit")
                    try:
                        a, b, c = curve_fit(exponential,
                          blockings["block_length"], blockings[field+"_se"],
                          p0=(0.01, -1.0, -0.1))[0]
                        if a >= 0:
                            exp_fit[field+"_exp_fit"] = exponential(
                              blockings["block_length"].values, a, b, c)
                            exp_par[field] = [a, b, c]
                        elif verbose >= 1:
                            print("Exponential fit for field "
                              "'{0}' yielded negative ".format(field) +
                              "standard error, setting values to NaN")
                    except RuntimeError:
                        if verbose >= 1:
                            print("Could not fit exponential for field "
                              "'{0}', setting values to NaN".format(field))
                if fit_sig:
                    columns.append(field+"_sig_fit")
                    try:
                        a, b, c, d = curve_fit(sigmoid,
                          blockings["n_transforms"], blockings[field+"_se"],
                          p0=(0.1, 0.1, 10, 1))[0]
                        if b >= 0:
                            sig_fit[field+"_sig_fit"] = sigmoid(
                              blockings["n_transforms"].values, a, b, c, d)
                            sig_par[field] = [a, b, c, d]
                        elif verbose >= 1:
                            print("Sigmoidal fit for field "
                              "'{0}' yielded negative ".format(field) +
                              "standard error, setting values to NaN")
                    except RuntimeError:
                        if verbose >= 1:
                            print("Could not fit sigmoid for field "
                              "'{0}', setting values to NaN".format(field))

        # Organize and return
        if fit_exp and fit_sig:
            parameters = pd.concat([exp_par, sig_par], keys=["exp", "sig"])
        elif fit_exp:
            parameters = pd.concat([exp_par], keys=["exp"])
        elif fit_sig:
            parameters = pd.concat([sig_par], keys=["sig"])
        else:
            parameters = None

        if fit_exp:
            blockings = blockings.join(exp_fit)
        if fit_sig:
            blockings = blockings.join(sig_fit)
        blockings = blockings[columns]

        if verbose >= 1:
            print(parameters)

        return blockings, parameters

    @arg_or_attr("blockings", "parameters")
    def plot(self, blockings, parameters, verbose=1, debug=0,
        outfile="test.pdf", **kwargs):
        """
        Plots block average results using matplotlib.

        Arguments:
          verbose (int): Level of verbose output
          debug (int): Level of debug output
          kwargs (dict): Additional keyword arguments
        Returns:

        """
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        fields = [n[:-5] for n in blockings.columns.tolist()
          if n.endswith("_mean")]
        n_fields = len(fields)
        fit_exp = True
        fit_sig = True

        # Generate and format figure and subplots
        figure, subplots = plt.subplots(n_fields, 2,
          figsize=[6.5, 2+n_fields*1.5],
          subplot_kw=dict(autoscale_on = True))
#        if self.n_fields == 1:
#            subplots = np.expand_dims(subplots, 0)
        # Must adjust for 1 or two column
        figure.tight_layout(pad=2, h_pad=-1, w_pad=-1)
#        figure.subplots_adjust(
#          left   = 0.10, wspace = 0.1, right = 0.95,
#          bottom = 0.06, hspace = 0.1, top   = 0.95)
#        figure.suptitle(self.name)
        # Title columns for sigmoid and exponential fit
        for i, field in enumerate(fields):
            # Format x
            if i != n_fields - 1:
                subplots[i,0].set_xticklabels([])
                subplots[i,1].set_xticklabels([])

            # Format y
            subplots[i,0].set_ylabel("σ", rotation="horizontal")
            subplots[i,1].set_yticklabels([])

            # Add y2 label
            subplots[i,1].yaxis.set_label_position("right")
            subplots[i,1].set_ylabel(field.title(), rotation=270, labelpad=15)


            # set xticks and yticks
        subplots[n_fields-1,0].set_xlabel("Block Length")
        subplots[n_fields-1,1].set_xlabel("Number of Block Transformations")

        for i, field in enumerate(fields):
            subplots[i,0].plot(
              blockings["block_length"], blockings[field+"_se"],
              color="blue")
            subplots[i,1].plot(
              blockings["n_transforms"], blockings[field+"_se"],
              color="blue")
            se_sd = blockings[field+"_se_sd"]
            subplots[i,0].fill_between(blockings["block_length"], 
              blockings[field+"_se"] - 1.96 * blockings[field+"_se_sd"],
              blockings[field+"_se"] + 1.96 * blockings[field+"_se_sd"],
              lw=0, alpha=0.5, color="blue")
            subplots[i,1].fill_between(blockings["n_transforms"],
              blockings[field+"_se"] - 1.96 * blockings[field+"_se_sd"],
              blockings[field+"_se"] + 1.96 * blockings[field+"_se_sd"],
              lw=0, alpha=0.5, color="blue")
            if fit_exp:
                subplots[i,0].plot(
                  blockings["block_length"], blockings[field+"_exp_fit"],
                  color="red")
#                subplots[i,1].plot(
#                  blockings["n_transforms"], blockings[field+"_exp_fit"],
#                  color="red")
            if fit_sig:
#                subplots[i,0].plot(
#                  blockings["block_length"], blockings[field+"_sig_fit"],
#                  color="green")
                subplots[i,1].plot(
                  blockings["n_transforms"], blockings[field+"_sig_fit"],
                  color="green")
        # Annotate
#                subplots[i,1].legend(loc = 4)

        # Also make sure y lower bound is 0 and x upper bound is max x
        # Scale exponential tick labels?
        for i, field in enumerate(fields):
            print(field)
            # Adjust x ticks
            xticks = subplots[i,0].get_xticks()
            xticks = xticks[xticks <= blockings["block_length"].max()]
            subplots[i,0].set_xbound(0, xticks[-1])
            subplots[i,1].set_xbound(0, blockings["n_transforms"].max())

            # Adjust y ticks
            yticks = subplots[i,0].get_yticks()
            yticks = yticks[yticks >= 0]
            subplots[i,0].set_ybound(yticks[0], yticks[-1])
            subplots[i,1].set_ybound(yticks[0], yticks[-1])

        # Save and return
        with PdfPages(outfile) as pdf_outfile:
            figure.savefig(pdf_outfile, format="pdf")
        return None

    def main(self, parser=None):
        """
        Provides command-line functionality.

        Arguments:
          parser (ArgumentParser, optional): argparse argument parser;
            enables sublass to instantiate parser and add arguments;
            feature not well tested
        """
        import argparse
        from inspect import getmodule

        if parser is None:
            parser = argparse.ArgumentParser(
              description     = getmodule(self.__class__).__doc__,
              formatter_class = argparse.RawDescriptionHelpFormatter)

        parser.add_argument(
          "-infile", "-infiles",
          type     = str,
          nargs    = "+",
          action   = "append",
          required = True,
          help     = "Input file(s)")

        parser.add_argument(
          "-max-cut",
          type     = float,
          dest     = "max_cut",
          default  = 0.1,
          help     = "Maximum proportion of dataset to cut when blocking; for "
                     "example, a dataset of length 21 might be divided into 2 "
                     "blocks of 10 (0.05 cut), 4 blocks of 5 (0.05 cut), 8 "
                     "blocks of 2 (0.24 cut), or 16 blocks of 1 (0.24 cut); "
                     "the latter two points would be used only if max_cut is "
                     "greater than 0.24. Default: %(default)s")

        parser.add_argument(
          "-min-n-blocks",
          type     = int,
          dest     = "min_n_blocks",
          default  = 1,
          help     = "Only use blockings that include at least this number "
                     "of blocks. Default: %(default)s")

        parser.add_argument(
          "--all-factors",
          action   = "store_true",
          dest     = "all_factors",
          help     = "Divide dataset into 2,3,4,5,... blocks rather than "
                     "2,4,8,16,... blocks; recommended for testing only")

#        parser.add_argument(
#          "-outfile",
#          type     = str,
#          nargs    = "+",
#          action   = "append",
#          help     = "Output results to file")
#
#        parser.add_argument(
#          "-outfigure",
#          type     = str,
#          default  = "block_average.pdf",
#          help     = "Output figure file (default: %(default)s)")

        parser.add_argument(
          "-s",
          "--seaborn",
          action   = "store_const",
          const    = 1,
          default  = 0,
          help     = "Enable seaborn, overriding matplotlib defaults")

        verbosity = parser.add_mutually_exclusive_group()

        verbosity.add_argument(
          "-v",
          "--verbose",
          action   = "count",
          default  = 1,
          help     = "Enable verbose output, may be specified more than once")

        verbosity.add_argument(
          "-q",
          "--quiet",
          action   = "store_const",
          const    = 0,
          default  = 1,
          dest     = "verbose",
          help     = "Disable verbose output")

        parser.add_argument(
          "-d",
          "--debug",
          action   = "count",
          default  = 0,
          help     = "Enable debug output, may be specified more than once")

        arguments = vars(parser.parse_args())

        if arguments["seaborn"] == 1:
            import seaborn
            seaborn.set_palette("muted")

        if arguments["debug"] >= 1:
            from os import environ
            from .debug import db_s, db_kv

            db_s("Environment variables")
            for key in sorted(environ):
                db_kv(key, environ[key], 1)

            db_s("Command-line arguments")
            for key in sorted(arguments.keys()):
                db_kv(key, arguments[key], 1)

        self(**arguments)

#################################### MAIN #####################################
if __name__ == "__main__":
    FPBlockAverager().main()

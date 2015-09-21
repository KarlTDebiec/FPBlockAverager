#!/usr/bin/env python
# -*- coding: utf-8 -*-
#   fpblockaverager.FPBlockAverager.py
#
#   Copyright (C) 2014-2015 Karl T Debiec
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
from sys import exit
################################### CLASSES ###################################
class FPBlockAverager(object):
    """
    Class to manage estimation of standard error using the
    block-averaging method of Flyvbjerg and Petersen

    Flyvbjerg, H., and Petersen, H. G. Error estimates on averages of
    correlated data. Journal of Chemical Physics. 1989. 91 (1). 461-466.
    """
    def __init__(self, **kwargs):
        """
        Initializes.

        Arguments:
          dataset (DataFrame): Dataset
          name (str): Name of dataset
          fieldnames (list): Name of fields of dataset
          full_length (int): Full length of dataset
          factor (int): Factor by which all block sizes must be
            divisible
          min_n_blocks (int): Minimum block size for transformations
          max_omitted (float): Maximum proportion of original dataset
            that may be omitted from end of block-transformed dataset
          verbose (int): Level of verbose output
          debug (int): Level of debug output
          kwargs (dict): Additional keyword arguments

        .. todo:
          - Fix
          - Clean up debug; looks embarrisingly outdated
          - Add example data sets and plots
          - Load from hdf5 using pandas or otherwise
        """
        from time import strftime

#        self.dataset      = dataset
#        self.full_length  = kwargs.pop("full_length", dataset.shape[0])
#        if "n_fields" in kwargs:
#            self.n_fields = kwargs.pop("n_fields")
#        elif len(dataset.shape) > 1:
#            self.n_fields = dataset.shape[1]
#        else: 
#            self.n_fields = 1
#        self.name         = kwargs.pop("name", strftime("%Y-%m-%d %H:%M:%S"))
#        self.fieldnames   = kwargs.pop("fieldnames", range(self.n_fields))
#        self.factor       = factor
#        self.min_n_blocks = min_n_blocks
#        self.max_omitted  = max_omitted
#        self.debug        = debug
#
#        self.full_length  = self.full_length - (self.full_length % factor)

    def __call__(self, **kwargs):
        """
        Carries out standard error calculation.

        Arguments:
          kwargs (dict): Additional keyword arguments
        """
        dataset = self.load_datasets(**kwargs)
        blockings = self.select_blockings(dataset=dataset, **kwargs)
        print(dataset)
        print(blockings)
#        self.calculate_blocks(**kwargs)
#        self.fit_curves(**kwargs)
#        if self.debug:
#            self.plot()

    def load_datasets(self, infile, **kwargs):
        """
        Load datasets from text, numpy, or hdf formats.
        """
        import pandas

        dataset = pandas.read_hdf(infile[0][0], infile[0][1])

        return dataset

    def select_blockings(self, dataset=None, max_cut=0.1,
        all_factors=False, **kwargs):
        """
        Selects lengths of block-transformed datasets.

        Arguments:
          kwargs (dict): Additional keyword arguments
        """
        import pandas
        import numpy as np

        if dataset is None:
            if hasattr(self, "dataset"):
                dataset = self.dataset
            else:
                raise ValueError("FPBlockAverager.select_blockings could "
                  "not identify dataset.")

        full_length = dataset.shape[0]

        if all_factors:
            # Must remove duplicates
            n_blocks = np.array(sorted(set(
              range(1, 2 ** int(np.floor(np.log2(full_length)))))), np.int)
            block_lengths = np.array(sorted(set(
              np.array([full_length / n for n in n_blocks], np.int))))[::-1]
            n_blocks = np.array(full_length / block_lengths, np.int)
        else:
            n_blocks = np.array([2 ** i for i in
                         range(int(np.floor(np.log2(full_length))))], np.int)
            block_lengths = np.array([full_length / n for n in n_blocks],
                              np.int)
        total_lengths = n_blocks * block_lengths
        n_transforms  = np.log2(n_blocks)

        # Cut blockings fot which more than max_cut proprotion of
        # dataset must be omitted
        max_cut_indexes = np.where(total_lengths/full_length >= 1-max_cut)[0]
        n_blocks = n_blocks[max_cut_indexes]
        block_lengths = block_lengths[max_cut_indexes]
        total_lengths = total_lengths[max_cut_indexes]
        n_transforms = n_transforms[max_cut_indexes]

        blockings = pandas.DataFrame(
          np.column_stack((
          n_transforms, n_blocks, block_lengths, total_lengths)),
          columns=["n transforms", "n blocks", "block length",
          "total length used"])

        return blockings

    def calculate_blocks(self, **kwargs):
        """
        Calculates standard error for each block transform.

        .. note:: 
            The standard deviation of each standard error
            (stderr_stddev) is only valid for points whose standard
            error has leveled off (i.e. can be assumed Gaussian).

        Arguments:
          kwargs (dict): Additional keyword arguments
        """
        self.means           = np.zeros((self.block_lengths.size,
                                         self.n_fields), np.float)
        self.stderrs         = np.zeros(self.means.shape, np.float)
        self.stderrs_stddevs = np.zeros(self.means.shape, np.float)

        for transform_i, block_length, n_blocks, total_length in zip(
          range(self.block_lengths.size), self.block_lengths, self.n_blocks,
          self.total_lengths):
            transformed   = self.transform(block_length, n_blocks,
                              total_length, **kwargs)
            mean          = np.mean(transformed, axis = 0)
            stddev        = np.std(transformed,  axis = 0)
            stderr        = stddev / np.sqrt(n_blocks - 1)
            stderr_stddev = stderr / np.sqrt(2 * (n_blocks - 1))
            self.means[transform_i,:]           = mean
            self.stderrs[transform_i,:]         = stderr
            self.stderrs_stddevs[transform_i,:] = stderr_stddev

        self.debug_block = ["{0:>12.5f} {1:>12.5f} {2:>12.5f}".format(
          a, b, c) for a, b, c in zip(self.means[:,0],
          self.stderrs[:,0], self.stderrs_stddevs[:,0])]
        if self.debug:
            self.print_debug()

    def transform(self, block_length, n_blocks, total_length, **kwargs):
        """
        Prepares a block-transformed dataset.

        Arguments:
          block_length (int): Length of each block in transformed
            dataset
          n_blocks (int): Number of blocks in transformed dataset 
          total_length (int): Number of frames in transformed dataset
          kwargs (dict): Additional keyword arguments
        """
        transformed = np.zeros((n_blocks, self.n_fields), np.float)
        for i in range(transformed.shape[1]):
            reshaped = np.reshape(self.dataset[:total_length, i], 
                         (n_blocks, block_length))
            transformed[:,i] = np.mean(reshaped, axis = 1)
        return transformed

    def fit_curves(self, **kwargs):
        """
        Fits exponential and sigmoid curves to block-transformed data.

        Arguments:
          kwargs (dict): Additional keyword arguments; passed to
            scipy.optimize.curve_fit
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

        self.exp_fit = np.zeros((self.n_transforms.size, self.n_fields))
        self.sig_fit = np.zeros((self.n_transforms.size, self.n_fields))
        self.exp_fit_parameters = np.zeros((3, self.n_fields))
        self.sig_fit_parameters = np.zeros((4, self.n_fields))

        with warnings.catch_warnings():
            for i in range(self.n_fields):
                try:
                    warnings.simplefilter("ignore")
                    a, b, c = curve_fit(exponential, self.block_lengths,
                      self.stderrs[:,i], p0 = (0.01, -1.0, -0.1), **kwargs)[0]
                    self.exp_fit[:,i] = exponential(self.block_lengths,a,b,c)
                    self.exp_fit_parameters[:,i] = [a, b, c]
                except RuntimeError:
                    warnings.simplefilter("always")
                    warnings.warn("Could not fit exponential for field "
                      "{0}, setting values to NaN".format(i))
                    self.exp_fit[:,i] = np.nan
                    self.exp_fit_parameters[:,i] = [np.nan,np.nan,np.nan]
                try:
                    warnings.simplefilter("ignore")
                    a, b, c, d = curve_fit(sigmoid, self.n_transforms,
                      self.stderrs[:,i], p0 = (0.1, 0.1, 10, 1), **kwargs)[0]
                    self.sig_fit[:,i] = sigmoid(self.n_transforms, a, b, c, d)
                    self.sig_fit_parameters[:,i] = [a, b, c, d]
                except RuntimeError:
                    warnings.simplefilter("always")
                    warnings.warn("Could not fit sigmoid for field "
                      "{0}, setting values to NaN".format(i))
                    self.sig_fit[:,i] = np.nan
                    self.sig_fit_parameters[:,i] = [np.nan,np.nan,np.nan,np.nan]

        self.debug_fit = ["{0:>12.5f} {1:>12.5f}".format(
          a, b) for a, b in zip(self.exp_fit[:,0], self.sig_fit[:,0])]
        self.debug_fit_parameters = "".join(["Exponential Fit:"] + 
          ["\n        "] +
          ["{0:>12}".format(str(fieldname)[:12])
            for fieldname in self.fieldnames] +
          ["\nA (SE)  "] +
          ["{0:>12.5f}".format(self.exp_fit_parameters[0,i])
            for i in range(self.n_fields)] +
          ["\nB       "] + 
          ["{0:>12.5f}".format(self.exp_fit_parameters[1,i])
            for i in range(self.n_fields)] +
          ["\nC       "] + 
          ["{0:>12.5f}".format(self.exp_fit_parameters[2,i])
            for i in range(self.n_fields)] +
          ["\nSigmoidal Fit:"] + 
          ["\n        "] + 
          ["{0:>12}".format(str(fieldname)[:12])
            for fieldname in self.fieldnames] +
          ["\nA       "] +
          ["{0:>12.5f}".format(self.sig_fit_parameters[0,i])
            for i in range(self.n_fields)] +
          ["\nB (SE)  "] + 
          ["{0:>12.5f}".format(self.sig_fit_parameters[1,i])
            for i in range(self.n_fields)] +
          ["\nC       "] + 
          ["{0:>12.5f}".format(self.sig_fit_parameters[2,i])
            for i in range(self.n_fields)] +
          ["\nD       "] + 
          ["{0:>12.5f}".format(self.sig_fit_parameters[3,i])
            for i in range(self.n_fields)])
        if self.debug:
            self.print_debug()

    def plot(self, **kwargs):
        """
        Plots block average results using matplotlib.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        outfile = kwargs.get("outfile", self.name.replace(" ", "_") + ".pdf")

        figure, subplots = plt.subplots(self.n_fields, 2,
          figsize = [11, 2 + self.n_fields * 3],
          subplot_kw = dict(autoscale_on = True))
        if self.n_fields == 1:
            subplots = np.expand_dims(subplots, 0)
        figure.subplots_adjust(
          left   = 0.10, wspace = 0.3, right = 0.97,
          bottom = 0.10, hspace = 0.4, top   = 0.95)
        figure.suptitle(self.name)
        
        for i in range(self.n_fields):
            subplots[i,0].set_title(self.fieldnames[i])
            subplots[i,1].set_title(self.fieldnames[i])
            subplots[i,0].set_xlabel("Block Length")
            subplots[i,1].set_xlabel("Number of Block Transformations")
            subplots[i,0].set_ylabel("$\sigma$")
            subplots[i,1].set_ylabel("$\sigma$")

            if hasattr(self, "stderrs"):
                subplots[i,0].plot(self.block_lengths, self.stderrs[:,i],
                  color = "blue")
                subplots[i,1].plot(self.n_transforms, self.stderrs[:,i],
                  color = "blue")
            if hasattr(self, "stderrs_stddevs"):
                subplots[i,0].fill_between(self.block_lengths,
                  self.stderrs[:,i] - 1.96 * self.stderrs_stddevs[:,i],
                  self.stderrs[:,i] + 1.96 * self.stderrs_stddevs[:,i],
                  lw = 0, alpha = 0.5, color = "blue")
                subplots[i,1].fill_between(self.n_transforms,
                  self.stderrs[:,i] - 1.96 * self.stderrs_stddevs[:,i],
                  self.stderrs[:,i] + 1.96 * self.stderrs_stddevs[:,i],
                  lw = 0, alpha = 0.5, color = "blue")
            if hasattr(self, "exp_fit"):
                if hasattr(self, "exp_fit_parameters"):
                    kwargs = dict(label = "SE = {0:4.2e}".format(
                               self.exp_fit_parameters[0,i]))
                else:
                    kwargs = {}
                subplots[i,0].plot(self.block_lengths, self.exp_fit[:,i],
                  color = "red", **kwargs)
                subplots[i,0].legend(loc = 4)
            if hasattr(self, "sig_fit"):
                if hasattr(self, "sig_fit_parameters"):
                    kwargs = dict(label = "SE = {0:4.2e}".format(
                               self.sig_fit_parameters[1,i]))
                else:
                    kwargs = {}
                subplots[i,1].plot(self.n_transforms, self.sig_fit[:,i],
                  color = "red", **kwargs)
                subplots[i,1].legend(loc = 4)
        with PdfPages(outfile) as pdf_outfile:
            figure.savefig(pdf_outfile, format = "pdf")
        print("Block average figure saved to '{0}'".format(outfile))

    def print_debug(self, length=True, block=True, fit=True,
          fit_parameters=True):
        """
        Prints debug information

        Outputs standard error and fits for first state only
        """
        if length and hasattr(self, "debug_length"):
            print("N_TRANSFORMS     N_BLOCKS BLOCK_LENGTH TOTAL_LENGTH",
              end = "")
        if block and hasattr(self, "debug_block"):
            print("        MEAN       STDERR    SE_STDDEV", end = "")
        if fit and hasattr(self, "debug_fit"):
            print("     EXP FIT     SIG FIT", end = "")
        print()

        for i in range(self.n_transforms.size):
            if length and hasattr(self, "debug_length"):
                print(self.debug_length[i], end = "")
            if block and hasattr(self, "debug_block"):
                print(self.debug_block[i], end = "")
            if fit and hasattr(self, "debug_fit"):
                print(self.debug_fit[i], end = "")
            print()
        if fit_parameters and hasattr(self, "debug_fit_parameters"):
            print(self.debug_fit_parameters)

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

#        parser.add_argument(
#          "-name",
#          type     = str,
#          help     = "Dataset name (default: current time)")
#
#        parser.add_argument(
#          "-outfile",
#          type     = str,
#          default  = "block_average.txt",
#          help     = "Output text file (default: %(default)s)")
#
#        parser.add_argument(
#          "-outfigure",
#          type     = str,
#          default  = "block_average.pdf",
#          help     = "Output figure file (default: %(default)s)")

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
          "--all-factors",
          action   = "store_true",
          dest     = "all_factors",
          help     = "Divide dataset into 2,3,4,5,... blocks rather than "
                     "2,4,8,16,... blocks; recommended for testing only")

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

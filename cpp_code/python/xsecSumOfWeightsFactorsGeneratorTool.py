# -*- coding: utf-8 -*-
import os
import re
from ROOT import TFile
from datetime import datetime

# Below is the only line you should need to change, add more directories containing sample directories as you wish
#input_sample_dirs = ["/eos/user/b/blumen/HWh", "/eos/user/a/adsalvad/public/HplusWhL2/Boosted/"] # Old ntuples
# input_sample_dirs = ["/eos/user/l/lubaines/ATLAS_SHARE/HpWh_L1ntuples/", "/eos/user/t/tqiu/H+Wh_ntuples/"] # New ntuples
input_sample_dirs = ["/data/atlas/HplusWh/20241021_RawNtuples/"] # New ntuples
# Assumes a file structure like /path/to/main/directory/sample_names/fileN.root
# where /path/to/main/directory/ is an entry in input_sample_dirs, sample_names are
# sample directories with a unique DSID/mcPeriod combination each, and fileN.root are
# a collection of root files (ntuples? or AODs?) for the given sample.
# We also assume that the DSID has the format <digit><digit><digit><digit><digit><digit>,
# and that the tags "r9364", "r10201", "r10724" refer to "MC16a", "MC16d", "MC16e" respectively
for i in range(2):
    print("Might be worth trying to speed this up by using TChains if you can!")
output_file = "../main/xSecSumOfWeightsFactors.h"
output_file = "tmp3.h"
with open(output_file, 'w') as outf:
    outf.write("// Created on %s using ../python/xsecSumOfWeightsFactorsGeneratorTool.py code\n" %(datetime.now().strftime("%d/%m/%Y at %H:%M:%S")))
    outf.write("#include <map>\n")
    outf.write("std::map<std::string,std::map<int,double>> xSecSumOfWeightsFactors;\n")

dsid_dict = {"MC16a" : {}, "MC16d" : {}, "MC16e" : {}}
for directory_in_str in input_sample_dirs:
    print("Checking directory: " + directory_in_str)
    directory = directory_in_str
    print("Looping through " + str(len(os.listdir(directory))) + " samples in this directory")
    for sample in os.listdir(directory):
        print("Checking sample: " + sample)
        # Check that is appears to be a sample name
        # if not (sample.startswith('user.rhulsken.mc16_13TeV.413') or sample.startswith('user.rhulsken.mc16_13TeV.412')):
        if not (sample.startswith('user.rhulsken.mc16_13TeV.411076')):
            continue
        if ("mc16" not in sample) and ("MC16" not in sample):
            print("mc16 doesn't appear in sample " + sample + "so I'm skipping this")
            continue
        # Get the DSID
        dsid_regex = re.search("\.[0-9]{6}\.", sample)
        if dsid_regex:
            dsid = dsid_regex.group()[1:-1]
            dsid_from_filename = True
            print("Found dsid: %s for sample %s" % (str(dsid), sample))
            #print("This will assume that we have the directory structure %s/%s/<filename> where each of <filename> is a file with the same dsid %d" %(directory, sample, dsid))
        else:
            print("Failed to find a string matching the pattern .<digit><digit><digit><digit><digit><digit>. in samplename " + sample + "; going to try and get the dsid by opening the file")
            print("WARNING: This will assume that we have the directory structure %s/%s/<filename> where each of <filename> is a file with it's own unique dsid" %(directory, sample))
            dsid_from_filename = False
        # Hack because of some weird specific directory structure for two particle files in the old ntuples
        if (sample == "MC16d_boost_ttbar_lep") or (sample == "MC16e_boost_ttbar_lep"):
            dsid = 410470
            dsid_from_filename = True
        # Now identify the MC Period
        if ("r9364" in sample) or ("MC16a" in sample):
            MC_period = "MC16a"
        elif ("r10201" in sample) or ("MC16d" in sample):
            MC_period = "MC16d"
        elif ("r10724" in sample) or ("MC16e" in sample):
            MC_period = "MC16e"
        else:
            print("WARNING: Failed to find a MC Data period tag in sample: " + sample)
        if not dsid_from_filename:
            # Now loop through and get the sum of all weights
            print("Looping through " + str(len(os.listdir(directory + "/" + sample))) + " files for this sample")
            for file in os.listdir(directory + "/" + sample):
                filepath = directory + "/" + sample + "/" + file
                #print("Opening file: " + filepath)
                try:
                    f = TFile.Open(filepath, "READ")
                except OSError:
                    print("Couldn't open file: %s so I'm assuming it's corrupted and I'm going to skip it" %(filepath))
                    continue
                try:
                    t = f.Get("nominal_Loose")
                    t.GetEntry(0)
                    dsid = t.mcChannelNumber
                    print("Found dsid: %s for sample/file %s/%s" % (str(dsid), sample, file))
                except ReferenceError:
                    print("Failed to get nominal_Loose TTree from file: " + filepath)
                    continue
                try:
                    t = f.Get("sumWeights")
                except ReferenceError:
                    print("Failed to get sumWeights TTree from file: " + filepath)
                    continue
                entries = t.GetEntries()
                #print("Looping through " + str(entries) + " entries in the file " + str(MC_period) + "," + str(dsid) + "," + file)
                sumWeights = 0
                for entry in range(entries):
                    t.GetEntry(entry)
                    sumWeights += t.totalEventsWeighted
                f.Close()
                if dsid in dsid_dict[MC_period].keys():
                    print("DUPLICATE WEIGHT!")
                dsid_dict[MC_period][dsid] = sumWeights
                print("xSecSumOfWeightsFactors[\"{}\"][{}]={};\n".format(MC_period, dsid, dsid_dict[MC_period][dsid]))
                with open(output_file, 'a') as outf:
                    outf.write("xSecSumOfWeightsFactors[\"{}\"][{}]={};\n".format(MC_period, dsid, dsid_dict[MC_period][dsid]))
        else:
            sumWeights = 0
            print("Looping through " + str(len(os.listdir(directory + "/" + sample))) + " files in this directory/sampleDirectory")
            for file in os.listdir(directory + "/" + sample):

                filepath = directory + "/" + sample + "/" + file
                #print("Opening file: " + filepath)
                f = TFile.Open(filepath, "READ")
                try:
                    t = f.Get("sumWeights")
                except ReferenceError:
                    print("Failed to get sumWeights TTree from file: " + filepath)
                    continue
                entries = t.GetEntries()
                #print("Looping through " + str(entries) + " entries in the file " + str(MC_period) + "," + str(dsid) + "," + file)
                for entry in range(entries):
                    t.GetEntry(entry)
                    sumWeights += t.totalEventsWeighted
                f.Close()
            #if dsid in dsid_dict.keys():
            #    print("DUPLICATE WEIGHT!")
            dsid_dict[MC_period][dsid] = sumWeights
            print("xSecSumOfWeightsFactors[\"{}\"][{}]={};\n".format(MC_period, dsid, dsid_dict[MC_period][dsid]))
            with open(output_file, 'a') as outf:
                outf.write("xSecSumOfWeightsFactors[\"{}\"][{}]={};\n".format(MC_period, dsid, dsid_dict[MC_period][dsid]))

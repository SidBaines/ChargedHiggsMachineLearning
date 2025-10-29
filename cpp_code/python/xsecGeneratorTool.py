from datetime import datetime

input_xsec_file_name="/eos/atlas/atlascerngroupdisk/asg-calib/dev/PMGTools/2022-02-21/PMGxsecDB_mc16.txt"
input_xsec_file_name="/eos/atlas/atlascerngroupdisk/asg-calib/dev/PMGTools/2023-03-02/PMGxsecDB_mc16.txt"
output_file="../main/xsec.h"
#output_file="../main/xsec.newntuples.h"
# Get dsids using eg: ll <data_path> | grep <some_identifier> | tr -s ' ' | cut -d ' ' -f 9 | cut -d '.' -f 4 | uniq
if 0:
    dsids = [410470,410471,410472,411073,411074,411075,411076,411077,411078,504558,504562,504565,504567,504569,504570,504571,
        364156, 364157,364158,364159,364160,364161,364162,364163,364164,364165,364166,364167,
        364168,364169,364170,364171,364172,364173,364174,364175,364176,
        364177,364178,364179,364180,364181,364182,364183,364184,364185,
        364186,364187,364188,364189,364190,364191,364192,364193,364194,
        364195,364196,364197
    ]
else:
    dsids = [
        363355,363356,363357,363358,363359,363360,363489,364184,364185,364186,364187,364188,364189,364190,364191,364192,364193,364194,364195,364196,364197,
        407342,407343,407344,407348,407349,410470,410646,410647,410654,410655,411073,411074,411075,411076,411077,411078,
        510115,510116,510117,510118,510119,510120,510121,510122,510123,510124,
        700320,700321,700322,700323,700324,700325,700326,700327,700328,700329,700330,700331,700332,700333,700334,
        700338,700339,700340,700341,700342,700343,700344,700345,700346,700347,700348,700349,
        413008, 413023, 412043,
    ]



dsids = [str(dsid) for dsid in dsids]

unfixed_dsids = [i for i in dsids]

with open(input_xsec_file_name, 'r') as file1:
    with open(output_file, 'w') as outf:
        outf.write("// Created on %s using ../python/xsecGeneratorTool.py code\n" %(datetime.now().strftime("%d/%m/%Y at %H:%M:%S")))
        outf.write("// xsecs & kfacs read from the file: %s\n" %(input_xsec_file_name))
        outf.write("#include <TString.h>\n")
        outf.write("#include <iostream>\n")
        outf.write("#include <vector>\n")
        outf.write("#include <map>\n")
        outf.write("// xsec in pb\n")
        outf.write("\n")
        outf.write("\n")
        outf.write("\n")
        outf.write("std::map<int,double> dsid_xsec;\n")
    
        Lines = file1.readlines()
        for line in Lines:
            line_list = line.split()
            if (line_list[0] in dsids):
                try:
                    unfixed_dsids.remove(line_list[0])
                except ValueError:
                    print("Trying to remove dsid %s from list of unfixed_dsids, failing, this likely implies a duplicate entry of this dsid in file %s" %(line_list[0], input_xsec_file_name))
                #print("dsid_xsec[{}]={};".format(line_list[0], float(line_list[2])*float(line_list[3])))
                outf.write("dsid_xsec[{}]={};\n".format(line_list[0], float(line_list[2])*float(line_list[3])))

with open(output_file, 'a') as outf:
    for dsid in unfixed_dsids:
        print("dsid_xsec[{}]={}; // Set to 1.0 since this dsid was not found in file {}".format(dsid, 1.0, input_xsec_file_name))
        outf.write("dsid_xsec[{}]={}; // Set to 1.0 since this dsid was not found in file {}\n".format(dsid, 1.0, input_xsec_file_name))
    print(" ")
    outf.write("\n")

with open(input_xsec_file_name, 'r') as file1:
    with open(output_file, 'a') as outf:
        outf.write("\n")
        outf.write("\n")
        outf.write("\n")
        outf.write("std::map<int,double> dsid_kfac;\n")
        Lines = file1.readlines()
        for line in Lines:
            line_list = line.split()
            if (line_list[0] in dsids):
                #print("dsid_kfac[{}]={};".format(line_list[0], float(line_list[4])))
                outf.write("dsid_kfac[{}]={};\n".format(line_list[0], float(line_list[4])))
with open(output_file, 'a') as outf:
    for dsid in unfixed_dsids:
        #print("dsid_kfac[{}]={}; // Set to 1.0 since this dsid was not found in file {}".format(dsid, 1.0, input_xsec_file_name))
        outf.write("dsid_kfac[{}]={}; // Set to 1.0 since this dsid was not found in file {}\n".format(dsid, 1.0, input_xsec_file_name))

if 0:
    print(" ")
    with open(input_xsec_file_name, 'r') as file1:
        Lines = file1.readlines()
        for line in Lines:
            line_list = line.split()
            if (line_list[0] in dsids):
                print(line_list)

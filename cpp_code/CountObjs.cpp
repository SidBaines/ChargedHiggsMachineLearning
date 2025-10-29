// Script to loop through a collection of root files, loop through the events in the tree, and fill a histogram with the number of objects
# include <TFile.h>
# include <TTree.h>
# include <TH1F.h>
# include <TString.h>
# include <string>
# include <fstream>
# include <iostream>
# include <vector>
# include <glob.h>
# include <sstream>
# include <algorithm>
# include <map>
# include <set>

// ALLOWED_SUBSTRINGS = 363355,363356,363357,363358,363359,363360,363489,407348,407349,410470,410646,410647,410654,410655,412043,413008,413023,510115,510116,510117,510118,510119,510120,510121,510122,510123,510124,700320,700321,700322,700323,700324,700325,700326,700327,700328,700329,700330,700331,700332,700333,700334,700338,700339,700340,700341,700342,700343,700344,700345,700346,700347,700348,700349


// Define allowed substrings
std::vector<std::string> allowed_substrings = {
    "363355", "363356", "363357", "363358", "363359", "363360", "363489",
    "407348", "407349", 
    "410470", 
    "410646", "410647", "410654", "410655",
    // "412043", 
    "413008", "413023", 
    "510115", "510116", "510117", "510118", "510119", "510120", "510121", "510122", "510123", "510124", 
    "700320", "700321", "700322", "700323", "700324", "700325", "700326", "700327",
    "700328", "700329", "700330", "700331", "700332", "700333", "700334",
    "700338", "700339", "700340", "700341", "700342", "700343", "700344",
    "700345", "700346", "700347", "700348", "700349"
};

// Function to check if filename contains allowed substring after "mc16_13TeV."
bool isFileAllowed(const std::string& filename) {
    
    // Find "mc16_13TeV." in the filename
    size_t pos = filename.find("mc16_13TeV.");
    if (pos == std::string::npos) {
        return false; // "mc16_13TeV." not found
    }
    
    // Extract substring starting after "mc16_13TeV."
    std::string after_mc16 = filename.substr(pos + 11); // 11 is length of "mc16_13TeV."
    
    // Check if any of the allowed substrings appears at the beginning of after_mc16
    for (const auto& allowed : allowed_substrings) {
        if (after_mc16.substr(0, allowed.length()) == allowed) {
            return true;
        }
    }
    
    return false;
}

// Function to create histograms for a given DSID
void createHistogramsForDSID(std::map<int, TH1F*>& h1_map, std::map<int, TH1F*>& h2_map, 
                            std::map<int, TH1F*>& h3_map, std::map<int, TH1F*>& h4_map, 
                            std::map<int, TH1F*>& h5_map, std::map<int, TH1F*>& h6_map, int dsid) {
    std::string dsid_str = std::to_string(dsid);
    
    h1_map[dsid] = new TH1F(("h1_DSID_" + dsid_str).c_str(), ("Number of objects (DSID " + dsid_str + ")").c_str(), 100, 0, 50);
    h2_map[dsid] = new TH1F(("h2_DSID_" + dsid_str).c_str(), ("Number of sjets (DSID " + dsid_str + ")").c_str(), 100, 0, 50);
    h3_map[dsid] = new TH1F(("h3_DSID_" + dsid_str).c_str(), ("Number of ljets (DSID " + dsid_str + ")").c_str(), 100, 0, 50);
    h4_map[dsid] = new TH1F(("h4_DSID_" + dsid_str).c_str(), ("Number of electrons (DSID " + dsid_str + ")").c_str(), 100, 0, 50);
    h5_map[dsid] = new TH1F(("h5_DSID_" + dsid_str).c_str(), ("Number of muons (DSID " + dsid_str + ")").c_str(), 100, 0, 50);
    h6_map[dsid] = new TH1F(("h6_DSID_" + dsid_str).c_str(), ("Number of neutrinos (DSID " + dsid_str + ")").c_str(), 100, 0, 50);
}

int main(int argc, char **argv)
{
    // Loop through files
    // std::string directory = "/data/atlas/HplusWh/20250313_WithTrueInclusion_FixedOverlapWHsjet_SmallJetCloseToLargeJetRemovalDeltaR0.5/";
    // std::string directory = "/data/atlas/HplusWh/20250305_WithTrueInclusion_FixedOverlapWHsjet/";
    std::string directory = "/data/atlas/HplusWh/20250115_SeparateLargeRJets_NominalWeights_extrainfo_fixed/";
    
    // Use glob to find .root files
    glob_t glob_result;
    std::string pattern = directory + "*.root";
    int ret = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    
    if (ret != 0) {
        std::cerr << "Error: No files found or glob failed" << std::endl;
        return 1;
    }
    
    std::vector<std::string> files;
    std::vector<std::string> filtered_files;
    
    // Collect all files first
    for (unsigned int i = 0; i < glob_result.gl_pathc; ++i) {
        files.push_back(std::string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    
    // Filter files based on allowed substrings
    for (const auto& filename : files) {
        if (isFileAllowed(filename)) {
            filtered_files.push_back(filename);
        }
    }
    
    std::cout << "Found " << files.size() << " total files, " << filtered_files.size() << " matching allowed patterns" << std::endl;
    
    if (filtered_files.empty()) {
        std::cerr << "Warning: No files match the allowed substring patterns" << std::endl;
        return 0;
    }
    
    // Maps to store histograms for each DSID
    std::map<int, TH1F*> h1_map; // Number of objects
    std::map<int, TH1F*> h2_map; // Number of sjets
    std::map<int, TH1F*> h3_map; // Number of ljets
    std::map<int, TH1F*> h4_map; // Number of electrons
    std::map<int, TH1F*> h5_map; // Number of muons
    std::map<int, TH1F*> h6_map; // Number of neutrinos

    std::set<int> encountered_dsids; // Keep track of DSIDs we've seen
    // Create histograms for all DSIDs (this is just the 'allowed substrings' list, but in integer format)
    for (const auto& dsid : allowed_substrings) {
        createHistogramsForDSID(h1_map, h2_map, h3_map, h4_map, h5_map, h6_map, std::stoi(dsid));
        encountered_dsids.insert(std::stoi(dsid));
    }
    
    
    // Loop through filtered files in directory
    for (size_t file_idx = 0; file_idx < filtered_files.size(); file_idx++)
    {
        const std::string& filename = filtered_files[file_idx];
        std::cout << "Processing file: " << filename << " (" << (file_idx+1) << " of " << filtered_files.size() << ")" << std::endl;
        
        // Open file
        TFile *rootfile = TFile::Open(filename.c_str());
        if (!rootfile || rootfile->IsZombie()) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            continue;
        }
        
        // Get tree
        TTree *tree = (TTree*)rootfile->Get("tree");
        if (!tree) {
            std::cerr << "Error: Cannot find tree in file " << filename << std::endl;
            rootfile->Close();
            continue;
        }
        
        // Create a vector and map the particle type branch to it
        std::vector<int> *particle_type = nullptr;
        float eventWeight, MET;
        int selection_category;
        uint DSID;
        
        tree->SetBranchStatus("*", 0);
        tree->SetBranchStatus("ll_particle_type", 1);
        tree->SetBranchStatus("eventWeight", 1);
        tree->SetBranchStatus("selection_category", 1);
        tree->SetBranchStatus("MET", 1);
        tree->SetBranchStatus("DSID", 1);
        
        tree->SetBranchAddress("ll_particle_type", &particle_type);
        tree->SetBranchAddress("eventWeight", &eventWeight);
        tree->SetBranchAddress("selection_category", &selection_category);
        tree->SetBranchAddress("MET", &MET);
        tree->SetBranchAddress("DSID", &DSID);
        
        // Loop through events in tree
        int num_objs = 0;
        int num_sjets = 0;
        int num_ljets = 0;
        int num_electrons = 0;
        int num_muons = 0;
        int num_neutrinos = 0;
        
        for (int i = 0; i < tree->GetEntries(); i++)
        {
            // Get event first
            tree->GetEntry(i);
            
            // Apply selection cuts
            if ((selection_category != 0) && (selection_category != 3) && (selection_category != 8) && (selection_category != 9) && (selection_category != 10)){
                continue;
            }
            if (MET < 30e3){
                continue;
            }
            
            // Check if particle_type is valid
            if (!particle_type) {
                continue;
            }
            
            // // Create histograms for this DSID if we haven't seen it before
            // if (encountered_dsids.find(DSID) == encountered_dsids.end()) {
            //     std::cout << "Creating histograms for DSID: " << DSID << std::endl;
            //     createHistogramsForDSID(h1_map, h2_map, h3_map, h4_map, h5_map, h6_map, DSID);
            //     encountered_dsids.insert(DSID);
            //     std::cout << "Created histograms for DSID: " << DSID << std::endl;
            // }
            
            // reset the counts
            num_objs = 0;
            num_sjets = 0;
            num_ljets = 0;
            num_electrons = 0;
            num_muons = 0;
            num_neutrinos = 0;
            
            num_objs = particle_type->size();
            for (size_t j = 0; j < particle_type->size(); j++)
            {
                if ((*particle_type)[j] == 4)
                {
                    num_sjets++;
                }
                if ((*particle_type)[j] == 3)
                {
                    num_ljets++;
                }
                if ((*particle_type)[j] == 0)
                {
                    num_electrons++;
                }
                if ((*particle_type)[j] == 1)
                {
                    num_muons++;
                }
                if ((*particle_type)[j] == 2)
                {
                    num_neutrinos++;
                }
            }
            
            // Fill histograms for this DSID
            h1_map[DSID]->Fill(num_objs, eventWeight);
            h2_map[DSID]->Fill(num_sjets, eventWeight);
            h3_map[DSID]->Fill(num_ljets, eventWeight);
            h4_map[DSID]->Fill(num_electrons, eventWeight);
            h5_map[DSID]->Fill(num_muons, eventWeight);
            h6_map[DSID]->Fill(num_neutrinos, eventWeight);
        }
        
        // Close the file
        rootfile->Close();
    }
    
    // Save histograms
    TFile *outfile = TFile::Open("/data/atlas/baines/histograms_by_dsid.root", "RECREATE");
    if (!outfile) {
        std::cerr << "Error: Cannot create output file" << std::endl;
        return 1;
    }
    
    std::cout << "\nSaving histograms for " << encountered_dsids.size() << " DSIDs:" << std::endl;
    for (const auto& dsid : encountered_dsids) {
        std::cout << "  DSID: " << dsid << std::endl;
        h1_map[dsid]->Write();
        h2_map[dsid]->Write();
        h3_map[dsid]->Write();
        h4_map[dsid]->Write();
        h5_map[dsid]->Write();
        h6_map[dsid]->Write();
    }
    
    outfile->Close();
    
    std::cout << "Histograms saved to histograms_by_dsid.root" << std::endl;
    
    return 0;
}
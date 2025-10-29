// Script to plot histograms with configurable DSID groupings using ROOT's THStack
#include <TFile.h>
#include <TH1F.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TStyle.h>
#include <THStack.h>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <algorithm>

// Structure to hold group configuration
struct Group {
    std::string name;
    std::vector<int> dsids;
    int color;
};

// Function to configure groups - modify this function to change groupings
std::vector<Group> configureGroups() {
    std::vector<Group> groups;
    
    // Group 1: 0.8TeV H+
    groups.push_back({"0.8TeV H+", {510115}, kRed});
    
    // Group 2: 3.0TeV H+
    groups.push_back({"3.0TeV H+", {510124}, kBlue});
    
    // // Group 3: Other H+ masses
    // groups.push_back({"Other H+ masses", {510116, 510117, 510118, 510119, 510120, 510121, 510122, 510123}, kOrange});
    
    // Group 4: SM bkg
    groups.push_back({"SM bkg", {363355,363356,363357,363358,363359,363360,363489,
                        407348,407349,410470,410646,410647,410654,410655,412043,413008,413023,
                        700320,700321,700322,700323,700324,
                700325,700326,700327,700328,700329,700330,700331,700332,700333,700334,
                700338,700339,700340,700341,700342,700343,700344,700345,700346,700347,
                700348,700349}, kOrange});
    
    return groups;
}

// Function to get all DSIDs present in the ROOT file
std::vector<int> getAvailableDSIDs(TFile* file) {
    std::vector<int> dsids;
    
    TList* keys = file->GetListOfKeys();
    if (!keys) return dsids;
    
    for (int i = 0; i < keys->GetSize(); i++) {
        std::string keyName = keys->At(i)->GetName();
        if (keyName.find("h1_DSID_") == 0) {
            size_t pos = keyName.find_last_of("_");
            if (pos != std::string::npos) {
                int dsid = std::stoi(keyName.substr(pos + 1));
                dsids.push_back(dsid);
            }
        }
    }
    
    std::sort(dsids.begin(), dsids.end());
    return dsids;
}

// Function to plot histograms for a given type using THStack
void plotHistogramType(TFile* file, const std::string& histType, const std::string& title, const std::vector<Group>& groups, const std::vector<int>& availableDSIDs) {
    
    // Create canvas
    TCanvas* canvas = new TCanvas(("canvas_" + histType).c_str(), title.c_str(), 800, 600);
    canvas->SetLogy();
    
    // Create stack
    THStack* stack = new THStack(("stack_" + histType).c_str(), title.c_str());
    
    // Create legend
    TLegend* legend = new TLegend(0.7, 0.7, 0.9, 0.9);
    legend->SetBorderSize(0);
    legend->SetFillStyle(0);
    
    // Process each group
    for (const auto& group : groups) {
        TH1F* groupHist = nullptr;
        
        // Sum histograms for this group
        for (int dsid : group.dsids) {
            if (std::find(availableDSIDs.begin(), availableDSIDs.end(), dsid) == availableDSIDs.end()) {
                continue; // DSID not available
            }
            
            std::string histName = histType + "_DSID_" + std::to_string(dsid);
            TH1F* hist = (TH1F*)file->Get(histName.c_str());
            
            if (!hist) {
                std::cout << "Warning: Histogram " << histName << " not found!" << std::endl;
                continue;
            }
            
            if (!groupHist) {
                // First histogram for this group
                std::string groupHistName = histType + "_" + group.name;
                std::replace(groupHistName.begin(), groupHistName.end(), ' ', '_');
                groupHist = (TH1F*)hist->Clone(groupHistName.c_str());
                groupHist->SetTitle(group.name.c_str());
            } else {
                // Add subsequent histograms
                groupHist->Add(hist);
            }
        }
        
        if (groupHist) {
            // Set style
            groupHist->SetFillColor(group.color);
            groupHist->SetLineColor(group.color);
            groupHist->SetLineWidth(1);
            
            // Add to stack and legend
            stack->Add(groupHist);
            legend->AddEntry(groupHist, group.name.c_str(), "f");
        }
    }
    
    // Draw the stack
    if (stack->GetNhists() > 0) {
        stack->Draw("HIST");
        stack->GetXaxis()->SetTitle("Count");
        stack->GetYaxis()->SetTitle("Events");
        
        legend->Draw();
        
        // Save as PDF
        std::string outputName = histType + "_stacked.pdf";
        canvas->SaveAs(outputName.c_str());
        std::cout << "Saved plot: " << outputName << std::endl;
    } else {
        std::cout << "No histograms to plot for " << histType << std::endl;
    }
    
    delete canvas;
    delete legend;
    // THStack and histograms will be cleaned up automatically
}

int main(int argc, char** argv) {
    std::string inputFile = "/data/atlas/baines/histograms_by_dsid.root";
    
    if (argc > 1) {
        inputFile = argv[1];
    }
    
    TFile* file = TFile::Open(inputFile.c_str());
    if (!file || file->IsZombie()) {
        std::cerr << "Error: Cannot open file " << inputFile << std::endl;
        return 1;
    }
    
    // Set ROOT style
    gStyle->SetOptStat(0);
    // gStyle->SetPadGridX(true);
    // gStyle->SetPadGridY(true);
    
    // Get available DSIDs
    std::vector<int> availableDSIDs = getAvailableDSIDs(file);
    std::cout << "Available DSIDs in file: ";
    for (int dsid : availableDSIDs) {
        std::cout << dsid << " ";
    }
    std::cout << std::endl;
    
    // Configure groups
    std::vector<Group> groups = configureGroups();
    
    // Plot each histogram type
    plotHistogramType(file, "h1", "Number of Objects", groups, availableDSIDs);
    plotHistogramType(file, "h2", "Number of Small Jets", groups, availableDSIDs);
    plotHistogramType(file, "h3", "Number of Large Jets", groups, availableDSIDs);
    plotHistogramType(file, "h4", "Number of Electrons", groups, availableDSIDs);
    plotHistogramType(file, "h5", "Number of Muons", groups, availableDSIDs);
    plotHistogramType(file, "h6", "Number of Neutrinos", groups, availableDSIDs);
    
    file->Close();
    
    std::cout << "All stacked plots have been generated!" << std::endl;
    
    return 0;
} 
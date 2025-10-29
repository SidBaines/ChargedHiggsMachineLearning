// Script to plot histograms with configurable DSID groupings
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
#include <TColor.h>

// Structure to hold group configuration
struct Group {
    std::string name;
    std::vector<int> dsids;
    int color;
    int style;
};

// Function to configure groups - modify this function to change groupings
std::vector<Group> configureGroups() {
    std::vector<Group> groups;
    
    // Group 1: 0.8TeV H+
    groups.push_back({"0.8TeV H+", {510115}, kRed, 1});
    
    // Group 2: 3.0TeV H+
    groups.push_back({"3.0TeV H+", {510124}, kBlue, 1});
    
    // Group 3: ttbar (if you add it back to allowed substrings)
    // groups.push_back({"ttbar", {410470}, kGreen+2, 1});
    
    // Group 4: Other H+ masses
    groups.push_back({"Other H+ masses", {510116, 510117, 510118, 510119, 510120, 510121, 510122, 510123}, kOrange, 1});
    
    return groups;
}

// Function to get all DSIDs present in the ROOT file
std::vector<int> getAvailableDSIDs(TFile* file) {
    std::vector<int> dsids;
    
    // Get list of keys in the file
    TList* keys = file->GetListOfKeys();
    if (!keys) return dsids;
    
    // Extract DSID from histogram names (format: h1_DSID_12345)
    for (int i = 0; i < keys->GetSize(); i++) {
        std::string keyName = keys->At(i)->GetName();
        if (keyName.find("h1_DSID_") == 0) {  // Only look at h1 histograms to avoid duplicates
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

// Function to create grouped histogram
TH1F* createGroupedHistogram(TFile* file, const std::string& histType, const Group& group, const std::vector<int>& availableDSIDs) {
    std::string groupHistName = histType + "_" + group.name;
    std::replace(groupHistName.begin(), groupHistName.end(), ' ', '_');
    std::replace(groupHistName.begin(), groupHistName.end(), '.', 'p');
    
    TH1F* groupHist = nullptr;
    bool firstHist = true;
    
    for (int dsid : group.dsids) {
        // Check if this DSID is available in the file
        if (std::find(availableDSIDs.begin(), availableDSIDs.end(), dsid) == availableDSIDs.end()) {
            std::cout << "Warning: DSID " << dsid << " not found in file, skipping..." << std::endl;
            continue;
        }
        
        std::string histName = histType + "_DSID_" + std::to_string(dsid);
        TH1F* hist = (TH1F*)file->Get(histName.c_str());
        
        if (!hist) {
            std::cout << "Warning: Histogram " << histName << " not found!" << std::endl;
            continue;
        }
        
        if (firstHist) {
            groupHist = (TH1F*)hist->Clone(groupHistName.c_str());
            groupHist->SetTitle((group.name + " - " + groupHist->GetTitle()).c_str());
            firstHist = false;
        } else {
            groupHist->Add(hist);
        }
    }
    
    if (groupHist) {
        groupHist->SetLineColor(group.color);
        groupHist->SetMarkerColor(group.color);
        groupHist->SetLineStyle(group.style);
        groupHist->SetLineWidth(2);
        groupHist->SetMarkerStyle(20);
        groupHist->SetMarkerSize(0.8);
    }
    
    return groupHist;
}

// Function to create "others" group from remaining DSIDs
TH1F* createOthersHistogram(TFile* file, const std::string& histType, const std::vector<Group>& groups, const std::vector<int>& availableDSIDs) {
    // Collect all DSIDs that are already in defined groups
    std::vector<int> usedDSIDs;
    for (const auto& group : groups) {
        usedDSIDs.insert(usedDSIDs.end(), group.dsids.begin(), group.dsids.end());
    }
    
    // Find DSIDs that are not in any group
    std::vector<int> otherDSIDs;
    for (int dsid : availableDSIDs) {
        if (std::find(usedDSIDs.begin(), usedDSIDs.end(), dsid) == usedDSIDs.end()) {
            otherDSIDs.push_back(dsid);
        }
    }
    
    if (otherDSIDs.empty()) {
        return nullptr;
    }
    
    std::cout << "Creating 'Others' group with DSIDs: ";
    for (int dsid : otherDSIDs) {
        std::cout << dsid << " ";
    }
    std::cout << std::endl;
    
    Group othersGroup = {"Others", otherDSIDs, kGray+2, 2};
    return createGroupedHistogram(file, histType, othersGroup, availableDSIDs);
}

// Function to plot histograms for a given type
void plotHistogramType(TFile* file, const std::string& histType, const std::string& title, const std::vector<Group>& groups, const std::vector<int>& availableDSIDs) {
    TCanvas* canvas = new TCanvas(("canvas_" + histType).c_str(), title.c_str(), 800, 600);
    canvas->SetLogy(); // Use log scale for y-axis
    
    TLegend* legend = new TLegend(0.7, 0.7, 0.9, 0.9);
    legend->SetBorderSize(0);
    legend->SetFillStyle(0);
    
    std::vector<TH1F*> histograms;
    double maxY = 0;
    
    // Create grouped histograms
    for (const auto& group : groups) {
        TH1F* groupHist = createGroupedHistogram(file, histType, group, availableDSIDs);
        if (groupHist) {
            histograms.push_back(groupHist);
            legend->AddEntry(groupHist, group.name.c_str(), "l");
            if (groupHist->GetMaximum() > maxY) {
                maxY = groupHist->GetMaximum();
            }
        }
    }
    
    // // Create "others" histogram
    // TH1F* othersHist = createOthersHistogram(file, histType, groups, availableDSIDs);
    // if (othersHist) {
    //     histograms.push_back(othersHist);
    //     legend->AddEntry(othersHist, "Others", "l");
    //     if (othersHist->GetMaximum() > maxY) {
    //         maxY = othersHist->GetMaximum();
    //     }
    // }
    
    if (histograms.empty()) {
        std::cout << "No histograms to plot for " << histType << std::endl;
        delete canvas;
        delete legend;
        return;
    }
    
    // Draw histograms
    bool firstDraw = true;
    for (auto* hist : histograms) {
        if (firstDraw) {
            hist->SetMaximum(maxY * 1.5);
            hist->SetMinimum(0.1); // For log scale
            hist->Draw("HIST");
            firstDraw = false;
        } else {
            hist->Draw("HIST SAME");
        }
    }
    
    legend->Draw();
    
    // Save the plot
    // std::string outputName = histType + "_grouped.png";
    // canvas->SaveAs(outputName.c_str());
    // std::cout << "Saved plot: " << outputName << std::endl;
    
    // Also save as PDF
    outputName = histType + "_grouped.pdf";
    canvas->SaveAs(outputName.c_str());
    
    delete canvas;
    delete legend;
}

int main(int argc, char** argv) {
    std::string inputFile = "/data/atlas/baines/histograms_by_dsid.root";
    
    // Allow user to specify input file
    if (argc > 1) {
        inputFile = argv[1];
    }
    
    // Open the ROOT file
    TFile* file = TFile::Open(inputFile.c_str());
    if (!file || file->IsZombie()) {
        std::cerr << "Error: Cannot open file " << inputFile << std::endl;
        return 1;
    }
    
    // Set ROOT style
    gStyle->SetOptStat(0);
    gStyle->SetPadGridX(true);
    gStyle->SetPadGridY(true);
    
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
    plotHistogramType(file, "h1", "Number of Objects - Grouped by DSID", groups, availableDSIDs);
    plotHistogramType(file, "h2", "Number of Small Jets - Grouped by DSID", groups, availableDSIDs);
    plotHistogramType(file, "h3", "Number of Large Jets - Grouped by DSID", groups, availableDSIDs);
    plotHistogramType(file, "h4", "Number of Electrons - Grouped by DSID", groups, availableDSIDs);
    plotHistogramType(file, "h5", "Number of Muons - Grouped by DSID", groups, availableDSIDs);
    plotHistogramType(file, "h6", "Number of Neutrinos - Grouped by DSID", groups, availableDSIDs);
    
    file->Close();
    
    std::cout << "All plots have been generated!" << std::endl;
    
    return 0;
} 
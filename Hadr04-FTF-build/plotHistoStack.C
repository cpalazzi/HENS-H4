#include <sstream>

void plotHistoStack(){
//gStyle->SetOptStat(111111); // draw statistics on plots,

auto hs = new THStack("hs","Times of nCapture");

TFile *f1 = TFile::Open("water100out.root");
TFile *f2 = TFile::Open("water200out.root");
TFile *f3 = TFile::Open("water300out.root");
TFile *f4 = TFile::Open("water400out.root");
TFile *f5 = TFile::Open("water500out.root");
TFile *f6 = TFile::Open("water600out.root");
TFile *f7 = TFile::Open("water700out.root");
TFile *f8 = TFile::Open("water800out.root");
TFile *f9 = TFile::Open("water900out.root");
TFile *f10 = TFile::Open("water1000out.root");

TH1D* hist100 = (TH1D*)f1->Get("8");
TFitResultPtr fit100 = hist100->Fit("expo","S");
TH1D* hist200 = (TH1D*)f2->Get("8");
TFitResultPtr fit200 = hist200->Fit("expo","S");
TH1D* hist300 = (TH1D*)f3->Get("8");
TFitResultPtr fit300 = hist300->Fit("expo","S");
TH1D* hist400 = (TH1D*)f4->Get("8");
TFitResultPtr fit400 = hist400->Fit("expo","S");
TH1D* hist500 = (TH1D*)f5->Get("8");
TFitResultPtr fit500 = hist500->Fit("expo","S");
TH1D* hist600 = (TH1D*)f6->Get("8");
TFitResultPtr fit600 = hist600->Fit("expo","S");
TH1D* hist700 = (TH1D*)f7->Get("8");
TFitResultPtr fit700 = hist700->Fit("expo","S");
TH1D* hist800 = (TH1D*)f8->Get("8");
TFitResultPtr fit800 = hist800->Fit("expo","S");
TH1D* hist900 = (TH1D*)f9->Get("8");
TFitResultPtr fit900 = hist900->Fit("expo","S");
TH1D* hist1000 = (TH1D*)f10->Get("8");
TFitResultPtr fit1000 = hist1000->Fit("expo","S");

hs->Add(hist100);
hs->Add(hist200);
hs->Add(hist300);
hs->Add(hist400);
hs->Add(hist500);
hs->Add(hist600);
hs->Add(hist700);
hs->Add(hist800);
hs->Add(hist900);
hs->Add(hist1000);

//Get Entries
int entries = ((TH1*)(hs->GetStack()->Last()))->GetEntries();
cout<<"\nEntries "<<entries<<"\n"; 
// Cast as string
string entries_str = std::to_string(entries);

// Create canvas
TCanvas *c1 = new TCanvas("Times of nCapture","Times of nCapture");
hs->Draw("nostack");
hs->GetXaxis()->SetTitle("t [ms]");

// Draw arrows on the canvas
TArrow arrow100(1.147,174.70,0.11, 179, 0.01,"|>"); 
arrow100.SetLineWidth(1); 
arrow100.DrawClone();
TArrow arrow1000(0.7,825,0.069, 875, 0.01,"|>"); 
arrow1000.SetLineWidth(1); 
arrow1000.DrawClone();
// Add arrow labels
TLatex text100(1.15,179,"100 MeV"); 
text100.DrawClone();
TLatex text1000(0.71,825,"1000 MeV"); 
text1000.DrawClone();



// Build and Draw a legend
TLegend *leg = new TLegend(0.68,0.7,0.99,0.94);
leg->DrawClone("Same");
TText *t = new TText(1.55,1050,Form("Entries = %d",entries));
t->SetTextFont(1);
t->Draw();


c1->Modified();

}


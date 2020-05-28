#include <sstream>

void plotHistoPads(){
//gStyle->SetOptStat(111111); // draw statistics on plots,

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

TF1 *myexp = new TF1("myexp","[0]*exp(-x/[1])");
myexp->SetParameter(1,1);

// Create canvas
TCanvas *c1 = new TCanvas("Times of nCapture","Times of nCapture");
c1->Divide(4,3);

c1->cd(1);
TH1D* hist100 = (TH1D*)f1->Get("8");
TFitResultPtr fit100 = hist100->Fit("myexp","S");
hist100->SetTitle("100 MeV");
c1->cd(2);
TH1D* hist200 = (TH1D*)f2->Get("8");
TFitResultPtr fit200 = hist200->Fit("myexp","S");
hist200->SetTitle("200 MeV");
c1->cd(3);
TH1D* hist300 = (TH1D*)f3->Get("8");
TFitResultPtr fit300 = hist300->Fit("myexp","S");
hist300->SetTitle("300 MeV");
c1->cd(4);
TH1D* hist400 = (TH1D*)f4->Get("8");
TFitResultPtr fit400 = hist400->Fit("myexp","S");
hist400->SetTitle("400 MeV");
c1->cd(5);
TH1D* hist500 = (TH1D*)f5->Get("8");
TFitResultPtr fit500 = hist500->Fit("myexp","S");
hist500->SetTitle("500 MeV");
c1->cd(6);
TH1D* hist600 = (TH1D*)f6->Get("8");
TFitResultPtr fit600 = hist600->Fit("myexp","S");
hist600->SetTitle("600 MeV");
c1->cd(7);
TH1D* hist700 = (TH1D*)f7->Get("8");
TFitResultPtr fit700 = hist700->Fit("myexp","S");
hist700->SetTitle("700 MeV");
c1->cd(8);
TH1D* hist800 = (TH1D*)f8->Get("8");
TFitResultPtr fit800 = hist800->Fit("myexp","S");
hist800->SetTitle("800 MeV");
c1->cd(9);
TH1D* hist900 = (TH1D*)f9->Get("8");
TFitResultPtr fit900 = hist900->Fit("myexp","S");
hist900->SetTitle("900 MeV");
c1->cd(10);
TH1D* hist1000 = (TH1D*)f10->Get("8");
TFitResultPtr fit1000 = hist1000->Fit("myexp","S");
hist1000->SetTitle("1000 MeV");

c1->Draw();

}


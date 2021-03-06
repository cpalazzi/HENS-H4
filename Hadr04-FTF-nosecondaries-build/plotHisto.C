{
  gROOT->Reset();
  
  // Draw histos filled by Geant4 simulation 
  //   
  //TFile f = TFile("Water_on.root");
  //... or ...
  TFile *f = TFile::Open("waterout.root");
  //TFile f = TFile("Hadr04.root");      
  TCanvas* c1 = new TCanvas("c1", "  ");
  c1->SetLogy(0);
  c1->cd();
  c1->Update();   
  
  TH1D* hist1 = (TH1D*)f->Get("1");
  hist1->Draw("HIST");
  
  TH1D* hist2 = (TH1D*)f->Get("2");
  hist2->Draw("HIST");
  
  TH1D* hist3 = (TH1D*)f->Get("3");
  hist3->Draw("HIST");
  
  TH1D* hist4 = (TH1D*)f->Get("4");
  hist4->Draw("HIST");
  
  TH1D* hist5 = (TH1D*)f->Get("5");
  hist5->Draw("HIST");
  
  TH1D* hist6 = (TH1D*)f->Get("6");
  hist6->Draw("HIST");
  
  TH1D* hist7 = (TH1D*)f->Get("7");
  hist7->Draw("HIST");

  //hist8
  TH1D* hist8 = (TH1D*)f->Get("8");
  hist8->Draw();

  TFitResultPtr r = hist8->Fit("expo","S");  // TFitResultPtr contains the TFitResult
  // TMatrixDSym cov = r->GetCovarianceMatrix();   //  to access the covariance matrix
  // Double_t chi2   = r->Chi2();                  // to retrieve the fit chi2
  Double_t slope   = r->Parameter(1); // retrieve the value for the parameter 1 which is a slope
  cout<<"\n Slope "<<slope;
  cout<<"Decay time "<<-1*TMath::Power(slope,-1);
  Double_t slope_err = r->ParError(1);             // retrieve the error for the parameter 1
  r->Print("Q");                                // print minimum information of fit including covariance matrix

  hist8->Draw(); 

  


}  

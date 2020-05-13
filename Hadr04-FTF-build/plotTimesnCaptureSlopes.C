void plotTimesnCaptureSlopes(){

// Open files
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
// Get curves and fits
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

// Get slope values
double slope100 = fit100->Parameter(1);
double slope200 = fit200->Parameter(1);
double slope300 = fit300->Parameter(1);
double slope400 = fit400->Parameter(1);
double slope500 = fit500->Parameter(1);
double slope600 = fit600->Parameter(1);
double slope700 = fit700->Parameter(1);
double slope800 = fit800->Parameter(1);
double slope900 = fit900->Parameter(1);
double slope1000 = fit1000->Parameter(1);
// Get slope errors
double slopeErr100 = fit100->ParError(1);
double slopeErr200 = fit200->ParError(1);
double slopeErr300 = fit300->ParError(1);
double slopeErr400 = fit400->ParError(1);
double slopeErr500 = fit500->ParError(1);
double slopeErr600 = fit600->ParError(1);
double slopeErr700 = fit700->ParError(1);
double slopeErr800 = fit800->ParError(1);
double slopeErr900 = fit900->ParError(1);
double slopeErr1000 = fit1000->ParError(1);

// Values and errors on axes
const int n_points=10;
double x_vals[n_points]= 
    {100,200,300,400,500,600,700,800,900,1000};
double y_vals[n_points]=
    {slope100,slope200,slope300,slope400,slope500,slope600,slope700,slope800,slope900,slope1000};
double y_errs[n_points]=
    {slopeErr100,slopeErr200,slopeErr300,slopeErr400,slopeErr500,slopeErr600,slopeErr700,slopeErr800,slopeErr900,slopeErr1000};

// Instance of the graph
TGraphErrors graph(n_points,x_vals,y_vals,nullptr,y_errs);
graph.SetTitle("nCapture Times Exponential Parameters; Neutron beam energy[MeV]; Slope");

// The canvas on which we'll draw the graph
auto mycanvas = new TCanvas();

//Draw the graph
graph.DrawClone("APE");

}

int main(){
    plotTimesnCaptureSlopes();
}
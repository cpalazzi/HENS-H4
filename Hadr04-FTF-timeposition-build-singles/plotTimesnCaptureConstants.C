void plotTimesnCaptureConstants(){

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

TF1 *myexp = new TF1("myexp","[0]*exp(-x/[1])");
myexp->SetParameter(1,1);

// Get curves and fits
// N option stops fits being drawn
TH1D* hist100 = (TH1D*)f1->Get("8");
TFitResultPtr fit100 = hist100->Fit("myexp","SN");
TH1D* hist200 = (TH1D*)f2->Get("8");
TFitResultPtr fit200 = hist200->Fit("myexp","SN");
TH1D* hist300 = (TH1D*)f3->Get("8");
TFitResultPtr fit300 = hist300->Fit("myexp","SN");
TH1D* hist400 = (TH1D*)f4->Get("8");
TFitResultPtr fit400 = hist400->Fit("myexp","SN");
TH1D* hist500 = (TH1D*)f5->Get("8");
TFitResultPtr fit500 = hist500->Fit("myexp","SN");
TH1D* hist600 = (TH1D*)f6->Get("8");
TFitResultPtr fit600 = hist600->Fit("myexp","SN");
TH1D* hist700 = (TH1D*)f7->Get("8");
TFitResultPtr fit700 = hist700->Fit("myexp","SN");
TH1D* hist800 = (TH1D*)f8->Get("8");
TFitResultPtr fit800 = hist800->Fit("myexp","SN");
TH1D* hist900 = (TH1D*)f9->Get("8");
TFitResultPtr fit900 = hist900->Fit("myexp","SN");
TH1D* hist1000 = (TH1D*)f10->Get("8");
TFitResultPtr fit1000 = hist1000->Fit("myexp","SN");

// Get constant values
double constant100 = fit100->Parameter(0);
double constant200 = fit200->Parameter(0);
double constant300 = fit300->Parameter(0);
double constant400 = fit400->Parameter(0);
double constant500 = fit500->Parameter(0);
double constant600 = fit600->Parameter(0);
double constant700 = fit700->Parameter(0);
double constant800 = fit800->Parameter(0);
double constant900 = fit900->Parameter(0);
double constant1000 = fit1000->Parameter(0);
// Get constant errors
double constantErr100 = fit100->ParError(0);
double constantErr200 = fit200->ParError(0);
double constantErr300 = fit300->ParError(0);
double constantErr400 = fit400->ParError(0);
double constantErr500 = fit500->ParError(0);
double constantErr600 = fit600->ParError(0);
double constantErr700 = fit700->ParError(0);
double constantErr800 = fit800->ParError(0);
double constantErr900 = fit900->ParError(0);
double constantErr1000 = fit1000->ParError(0);

// Values and errors on axes
const int n_points=10;
double x_vals[n_points]= 
    {100,200,300,400,500,600,700,800,900,1000};
double y_vals[n_points]=
    {constant100,constant200,constant300,constant400,constant500,constant600,constant700,constant800,constant900,constant1000};
double y_errs[n_points]=
    {constantErr100,constantErr200,constantErr300,constantErr400,constantErr500,constantErr600,constantErr700,constantErr800,constantErr900,constantErr1000};

// Instance of the graph
TGraphErrors graph(n_points,x_vals,y_vals,nullptr,y_errs);
graph.SetTitle("nCapture Exponential Constants; Neutron beam energy [MeV]; Constant");

// Make the plot aesthetically better
graph.SetMarkerStyle(kOpenCircle);
graph.SetMarkerColor(kBlue);
graph.SetMarkerSize(0.1);
graph.SetLineColor(kBlue);

// The canvas on which we'll draw the graph
TCanvas *c1 = new TCanvas;

//Draw the graph
graph.DrawClone("APE");

}

int main(){
    plotTimesnCaptureConstants();
}
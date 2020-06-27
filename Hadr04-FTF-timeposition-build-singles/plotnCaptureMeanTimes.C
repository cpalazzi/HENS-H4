void plotnCaptureMeanTimes(){

// Open files
TFile *f1 = TFile::Open("water10out.root");
TFile *f2 = TFile::Open("water20out.root");
TFile *f3 = TFile::Open("water30out.root");
TFile *f4 = TFile::Open("water40out.root");
TFile *f5 = TFile::Open("water50out.root");
TFile *f6 = TFile::Open("water60out.root");
TFile *f7 = TFile::Open("water70out.root");
TFile *f8 = TFile::Open("water80out.root");
TFile *f9 = TFile::Open("water90out.root");
TFile *f10 = TFile::Open("water100out.root");
TFile *f11 = TFile::Open("water200out.root");
TFile *f12 = TFile::Open("water300out.root");
TFile *f13 = TFile::Open("water400out.root");
TFile *f14 = TFile::Open("water500out.root");
TFile *f15 = TFile::Open("water600out.root");
TFile *f16 = TFile::Open("water700out.root");
TFile *f17 = TFile::Open("water800out.root");
TFile *f18 = TFile::Open("water900out.root");
TFile *f19 = TFile::Open("water1000out.root");

TF1 *myexp = new TF1("myexp","[0]*exp(-x/[1])");
myexp->SetParameter(1,1);


// Get curves and fits
// N option stops fits being drawn
TH1D* hist10 = (TH1D*)f1->Get("8");
TFitResultPtr fit10 = hist10->Fit("myexp","SN");
TH1D* hist20 = (TH1D*)f2->Get("8");
TFitResultPtr fit20 = hist20->Fit("myexp","SN");
TH1D* hist30 = (TH1D*)f3->Get("8");
TFitResultPtr fit30 = hist30->Fit("myexp","SN");
TH1D* hist40 = (TH1D*)f4->Get("8");
TFitResultPtr fit40 = hist40->Fit("myexp","SN");
TH1D* hist50 = (TH1D*)f5->Get("8");
TFitResultPtr fit50 = hist50->Fit("myexp","SN");
TH1D* hist60 = (TH1D*)f6->Get("8");
TFitResultPtr fit60 = hist60->Fit("myexp","SN");
TH1D* hist70 = (TH1D*)f7->Get("8");
TFitResultPtr fit70 = hist70->Fit("myexp","SN");
TH1D* hist80 = (TH1D*)f8->Get("8");
TFitResultPtr fit80 = hist80->Fit("myexp","SN");
TH1D* hist90 = (TH1D*)f9->Get("8");
TFitResultPtr fit90 = hist90->Fit("myexp","SN");
TH1D* hist100 = (TH1D*)f10->Get("8");
TFitResultPtr fit100 = hist100->Fit("myexp","SN");
TH1D* hist200 = (TH1D*)f11->Get("8");
TFitResultPtr fit200 = hist200->Fit("myexp","SN");
TH1D* hist300 = (TH1D*)f12->Get("8");
TFitResultPtr fit300 = hist300->Fit("myexp","SN");
TH1D* hist400 = (TH1D*)f13->Get("8");
TFitResultPtr fit400 = hist400->Fit("myexp","SN");
TH1D* hist500 = (TH1D*)f14->Get("8");
TFitResultPtr fit500 = hist500->Fit("myexp","SN");
TH1D* hist600 = (TH1D*)f15->Get("8");
TFitResultPtr fit600 = hist600->Fit("myexp","SN");
TH1D* hist700 = (TH1D*)f16->Get("8");
TFitResultPtr fit700 = hist700->Fit("myexp","SN");
TH1D* hist800 = (TH1D*)f17->Get("8");
TFitResultPtr fit800 = hist800->Fit("myexp","SN");
TH1D* hist900 = (TH1D*)f18->Get("8");
TFitResultPtr fit900 = hist900->Fit("myexp","SN");
TH1D* hist1000 = (TH1D*)f19->Get("8");
TFitResultPtr fit1000 = hist1000->Fit("myexp","SN");

// Get slope values
double slope10 = fit10->Parameter(1);
double slope20 = fit20->Parameter(1);
double slope30 = fit30->Parameter(1);
double slope40 = fit40->Parameter(1);
double slope50 = fit50->Parameter(1);
double slope60 = fit60->Parameter(1);
double slope70 = fit70->Parameter(1);
double slope80 = fit80->Parameter(1);
double slope90 = fit90->Parameter(1);
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
double slopeErr10 = fit10->ParError(1);
double slopeErr20 = fit20->ParError(1);
double slopeErr30 = fit30->ParError(1);
double slopeErr40 = fit40->ParError(1);
double slopeErr50 = fit50->ParError(1);
double slopeErr60 = fit60->ParError(1);
double slopeErr70 = fit70->ParError(1);
double slopeErr80 = fit80->ParError(1);
double slopeErr90 = fit90->ParError(1);
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
const int n_points=20;
double x_vals[n_points]= 
    {10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000};
double y_vals[n_points]=
    {slope10,slope20,slope30,slope40,slope50,slope60,slope70,slope80,slope90
    ,slope100,slope200,slope300,slope400,slope500,slope600,slope700,slope800,slope900,slope1000};
double y_errs[n_points]=
    {slopeErr10,slopeErr20,slopeErr30,slopeErr40,slopeErr50,slopeErr60,slopeErr70,slopeErr80,slopeErr90
    ,slopeErr100,slopeErr200,slopeErr300,slopeErr400,slopeErr500,slopeErr600,slopeErr700,slopeErr800,slopeErr900,slopeErr1000};

// Instance of the graph
TGraphErrors graph(n_points,x_vals,y_vals,nullptr,y_errs);
graph.SetTitle("nCapture Exponential Mean Decay Times; Neutron beam energy [MeV]; Mean Decay Time [ms]");

// Make the plot aesthetically better
graph.SetMarkerStyle(kOpenCircle);
graph.SetMarkerColor(kBlue);
graph.SetLineColor(kBlue);

// The canvas on which we'll draw the graph
TCanvas *c1 = new TCanvas;

//Draw the graph
graph.DrawClone("APE");

}

int main(){
    plotnCaptureMeanTimes();
}
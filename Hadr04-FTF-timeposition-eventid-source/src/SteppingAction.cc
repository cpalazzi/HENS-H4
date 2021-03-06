//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
/// \file SteppingAction.cc
/// \brief Implementation of the SteppingAction class
//
// 
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include "SteppingAction.hh"
#include "Run.hh"
#include "TrackingAction.hh"
#include "HistoManager.hh"

#include "G4RunManager.hh"

#include <iostream>
#include <fstream>
using namespace std;
                           
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SteppingAction::SteppingAction(TrackingAction* TrAct)
: G4UserSteppingAction(),fTrackingAction(TrAct)
{ }

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SteppingAction::~SteppingAction()
{ }

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void SteppingAction::UserSteppingAction(const G4Step* aStep)
{
  // count processes
  // 
  const G4StepPoint* endPoint = aStep->GetPostStepPoint();
  const G4VProcess* process   = endPoint->GetProcessDefinedStep();
  Run* run = static_cast<Run*>(
        G4RunManager::GetRunManager()->GetNonConstCurrentRun());
  run->CountProcesses(process);

  // incident neutron
  //
  if (aStep->GetTrack()->GetTrackID() == 1) { 
    G4double ekin  = endPoint->GetKineticEnergy();
    G4double trackl = aStep->GetTrack()->GetTrackLength();
    G4double time   = aStep->GetTrack()->GetLocalTime();           
    fTrackingAction->UpdateTrackInfo(ekin,trackl,time);
    G4AnalysisManager::Instance()->FillH1(7,ekin);
  }    

  // Times of nCapture
  G4String process_name=aStep->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName();

  if(!process_name.compare("nCapture")){
    G4double t_finish=aStep->GetPostStepPoint()->GetGlobalTime();
    //fTrackingAction->UpdateTrackInfo(t_finish);
    G4AnalysisManager::Instance()->FillH1(8,t_finish);
  }

  // Positions of nCapture
  if(!process_name.compare("nCapture")){
    G4double time_finish=aStep->GetPostStepPoint()->GetGlobalTime();
    G4ThreeVector position_finish=aStep->GetPostStepPoint()->GetPosition();
    G4int eventid=G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();
    ofstream positionsfile;
    positionsfile.open ("timepositions.csv", ios_base::app);
    positionsfile << time_finish << ",";
    positionsfile << position_finish[0] << ",";
    positionsfile << position_finish[1] << ",";
    positionsfile << position_finish[2] << ",";
    positionsfile << eventid <<"\n";

    positionsfile.close();
    //G4cout<<position_finish;
    //fTrackingAction->UpdateTrackInfo(t_finish);
    //G4AnalysisManager::Instance()->FillH1(9,position_finish);
  }

  /*if(!process_name.compare("nCapture")){
    auto analysisManager = G4AnalysisManager::Instance();
    G4ThreeVector position_finish=aStep->GetPostStepPoint()->GetPosition();
    analysisManager->FillNtupleDColumn(0,0, position_finish);
    analysisManager->AddNtupleRow(0);
    }*/
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......



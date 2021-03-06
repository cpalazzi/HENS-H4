#
# for neutronHP and particleHP
#
unsetenv G4NEUTRONHP_SKIP_MISSING_ISOTOPES 
unsetenv G4NEUTRONHP_DO_NOT_ADJUST_FINAL_STATE
unsetenv G4NEUTRONHP_USE_ONLY_PHOTONEVAPORATION
unsetenv G4NEUTRONHP_NEGLECT_DOPPLER
unsetenv G4NEUTRONHP_PRODUCE_FISSION_FRAGMENTS

unsetenv G4PHP_DO_NOT_ADJUST_FINAL_STATE
unsetenv G4PHP_NELECT_DOPPLER

#### setenv G4NEUTRONHP_SKIP_MISSING_ISOTOPES 1
#### setenv G4NEUTRONHP_DO_NOT_ADJUST_FINAL_STATE 1
#### setenv G4NEUTRONHP_USE_ONLY_PHOTONEVAPORATION 1
#### setenv G4NEUTRONHP_NELECT_DOPPLER 1
#### setenv G4NEUTRONHP_PRODUCE_FISSION_FRAGMENTS 1
#
#### setenv G4PHP_DO_NOT_ADJUST_FINAL_STATE 1
#### setenv G4PHP_NELECT_DOPPLER 1
#
# for Bertini cascade
#
unsetenv G4CASCADE_USE_PRECOMPOUND
#### setenv G4CASCADE_USE_PRECOMPOUND 1

env |grep G4NEUTRONHP
env |grep G4PHP
env |grep G4CASCADE

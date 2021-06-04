#include "neighborList.h"


void getNeighborsFromDescriptor(vector<vector<int > > &	d_nlist_a,
	     vector<vector<int > > &d_nlist_r, vector<compute_t> boxt, vector<compute_t>& positions, vector<int> types, double rcut_r, double rcut_a, double nloc, const SimulationRegion<compute_t> & region){
    // Define the return value.
    //vector<vector<int>> d_nlist_a;
    //vector<vector<int>> d_nlist_r;

    //SimulationRegion<compute_t> region;
    //region.reinitBox(boxt.data());

    vector<compute_t> bk_d_coord3 = positions;
    vector<int> bk_d_type = types;
    vector<int> ncell, ngcell;
    vector<int> nlist_map;
    copy_coord(positions, types, nlist_map, ncell, ngcell, bk_d_coord3, bk_d_type, rcut_r, region);
    bool b_nlist_map = true;
    vector<int> nat_stt(3, 0);
    vector<int> ext_stt(3), ext_end(3);
    for (int dd = 0; dd < 3; ++dd)
    {
        ext_stt[dd] = -ngcell[dd];
        ext_end[dd] = ncell[dd] + ngcell[dd];
        //cout<<"ext_end "<<dd<<" "<<ext_end[dd]<<endl;
    }
    ::build_nlist(d_nlist_a, d_nlist_r, positions, nloc, rcut_a, rcut_r, nat_stt, ncell, ext_stt, ext_end, region, ncell);
    cout<<positions.size()<<endl;
}




/**********************************************************************************************************************************
* Licensed under the MIT License (MIT) --- see LICENSE.txt
* Copyright © 2022 Niclas Boehmer (niclas.boehmer [at] tu-berlin.de) and Andrzej Kaczmarczyk (andrzej.kaczmarczyk [at] agh.edu.pl)
**********************************************************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;

#define BIG 100000
typedef int row;
#define ROW_TYPE INT
typedef int col;
#define COL_TYPE INT
typedef int cost;
#define COST_TYPE INT
#if !defined TRUE
#define	 TRUE		1
#endif
#if !defined FALSE
#define  FALSE		0
#endif
typedef int boolean;

/*This function is the jv shortest augmenting path algorithm to solve the assignment problem*/
cost lap(int dim,
        cost **assigncost,
        col *rowsol,
        row *colsol,
        cost *u,
        cost *v)

// input:
// dim        - problem size
// assigncost - cost matrix

// output:
// rowsol     - column assigned to row in solution
// colsol     - row assigned to column in solution
// u          - dual variables, row reduction numbers
// v          - dual variables, column reduction numbers

{
  boolean unassignedfound;
  row  i, imin, numfree = 0, prvnumfree, f, i0, k, freerow, *pred, *free;
  col  j, j1, endofpath, low, up, *collist, *matches;
  cost h, umin, usubmin, v2, *d;
  // avoid "uninitialized" warning
  cost min = 9999999;
  col last = -1;
  col j2 = -1;

  free = new row[dim];       // list of unassigned rows.
  collist = new col[dim];    // list of columns to be scanned in various ways.
  matches = new col[dim];    // counts how many times a row could be assigned.
  d = new cost[dim];         // 'cost-distance' in augmenting path calculation.
  pred = new row[dim];       // row-predecessor of column in augmenting/alternating path.

  // init how many times a row will be assigned in the column reduction.
  for (i = 0; i < dim; i++)
    matches[i] = 0;

  // COLUMN REDUCTION
  for (j = dim;j--;) // reverse order gives better results.
  {
    // find minimum cost over rows.
    min = assigncost[0][j];
    imin = 0;
    for (i = 1; i < dim; i++)
      if (assigncost[i][j] < min)
      {
        min = assigncost[i][j];
        imin = i;
      }
    v[j] = min;
    if (++matches[imin] == 1)
    {
      // init assignment if minimum row assigned for first time.
      rowsol[imin] = j;
      colsol[j] = imin;
    }
    else if(v[j]<v[rowsol[imin]]){
        int j1 = rowsol[imin];
        rowsol[imin] = j;
        colsol[j] = imin;
        colsol[j1] = -1;
    }
    else
      colsol[j] = -1;        // row already assigned, column not assigned.
  }

  // REDUCTION TRANSFER
  for (i = 0; i < dim; i++)
    if (matches[i] == 0)     // fill list of unassigned 'free' rows.
      free[numfree++] = i;
   else
      if (matches[i] == 1)   // transfer reduction from rows that are assigned once.  {
      {
        j1 = rowsol[i];
        min = BIG;
        for (j = 0; j < dim; j++)
          if (j != j1)
            if (assigncost[i][j] - v[j] < min)
              min = assigncost[i][j] - v[j];
        v[j1] = v[j1] - min;
      }

    //   AUGMENTING ROW REDUCTION
  int loopcnt = 0;           // do-loop to be done twice.
  do
  {
    loopcnt++;

    //     scan all free rows.
    //     in some cases, a free row may be replaced with another one to be scanned next.
    k = 0;
    prvnumfree = numfree;
    numfree = 0;             // start list of rows still free after augmenting row reduction.
    while (k < prvnumfree)
    {
      i = free[k];
      k++;

    //       find minimum and second minimum reduced cost over columns.
      umin = assigncost[i][0] - v[0];
      j1 = 0;
      usubmin = BIG;
      for (j = 1; j < dim; j++)
      {
        h = assigncost[i][j] - v[j];
        if (h < usubmin){
          if (h >= umin)
          {
            usubmin = h;
            j2 = j;
          }
          else
          {
            usubmin = umin;
            umin = h;
            j2 = j1;
            j1 = j;
          }
        }
      }

      i0 = colsol[j1];
      if (umin < usubmin){
    //         change the reduction of the minimum column to increase the minimum
    //         reduced cost in the row to the subminimum.
        v[j1] = v[j1] - (usubmin - umin);
      }
      else{                   // minimum and subminimum equal.
        if(i0 > -1)  // minimum column j1 is assigned.
        {
    //           swap columns j1 and j2, as j2 may be unassigned.
          j1 = j2;
          i0 = colsol[j2];
        }
      }

    //       (re-)assign i to j1, possibly de-assigning an i0.
      rowsol[i] = j1;
      colsol[j1] = i;

        if(i0 > -1){  // minimum column j1 assigned earlier.
            if (umin < usubmin){
        //           put in current k, and go back to that k.
        //           continue augmenting path i - j1 with i0.
                free[--k] = i0;
            }
            else{
        //           no further augmenting reduction possible.
        //           store i0 in list of free rows for next phase.
              free[numfree++] = i0;
            }
        }
    }
  }
  while (loopcnt < 2);       // repeat once.

  // AUGMENT SOLUTION for each free row.
  for (f = 0; f < numfree; f++)
  {
    freerow = free[f];       // start row of augmenting path.

    // Dijkstra shortest path algorithm.
    // runs until unassigned column added to shortest path tree.
    for(j = dim;j--;)
    {
      d[j] = assigncost[freerow][j] - v[j];
      pred[j] = freerow;
      collist[j] = j;        // init column list.
    }

    low = 0; // columns in 0..low-1 are ready, now none.
    up = 0;  // columns in low..up-1 are to be scanned for current minimum, now none.
             // columns in up..dim-1 are to be considered later to find new minimum,
             // at this stage the list simply contains all columns
    unassignedfound = FALSE;
    do
    {
      if (up == low)         // no more columns to be scanned for current minimum.
      {
        last = low - 1;

        // scan columns for up..dim-1 to find all indices for which new minimum occurs.
        // store these indices between low..up-1 (increasing up).
        min = d[collist[up++]];
        for (k = up; k < dim; k++)
        {
          j = collist[k];
          h = d[j];
          if (h <= min)
          {
            if (h < min)     // new minimum.
            {
              up = low;      // restart list at index low.
              min = h;
            }
            // new index with same minimum, put on undex up, and extend list.
            collist[k] = collist[up];
            collist[up++] = j;
          }
        }
        // check if any of the minimum columns happens to be unassigned.
        // if so, we have an augmenting path right away.
        for (k = low; k < up; k++)
          if (colsol[collist[k]] < 0)
          {
            endofpath = collist[k];
            unassignedfound = TRUE;
            break;
          }
      }

      if (!unassignedfound)
      {
        // update 'distances' between freerow and all unscanned columns, via next scanned column.
        j1 = collist[low];
        low++;
        i = colsol[j1];
        h = assigncost[i][j1] - v[j1] - min;

        for (k = up; k < dim; k++)
        {
          j = collist[k];
          v2 = assigncost[i][j] - v[j] - h;
          if (v2 < d[j])
          {
            pred[j] = i;
            if (v2 == min){  // new column found at same minimum value
              if (colsol[j] < 0)
              {
                // if unassigned, shortest augmenting path is complete.
                endofpath = j;
                unassignedfound = TRUE;
                break;
              }
              // else add to list to be scanned right away.
              else
              {
                collist[k] = collist[up];
                collist[up++] = j;
              }
            }
            d[j] = v2;
          }
        }
      }
    }
    while (!unassignedfound);

    // update column prices.
    for( k = last+1;k--;)
    {
      j1 = collist[k];
      v[j1] = v[j1] + d[j1] - min;
    }

    // reset row and column assignments along the alternating path.
    do
    {
      i = pred[endofpath];
      colsol[endofpath] = i;
      j1 = endofpath;
      endofpath = rowsol[i];
      rowsol[i] = j1;
    }
    while (i != freerow);
  }

  // calculate optimal cost.
  cost lapcost = 0;
//  for (i = 0; i < dim; i++)
  for(i = dim;i--;)
  {
    j = rowsol[i];
	u[i] = assigncost[i][j] - v[j];
    lapcost = lapcost + assigncost[i][j];
  }

   // free reserved memory.
  delete[] pred;
  delete[] free;
  delete[] collist;
  delete[] matches;
  delete[] d;
  return lapcost;
}

int getInvCount(int arr[], int arr2[], int n, bool swap)
{
    int inv_count = 0;
    if(swap) {
        for (int i = 0; i < n - 1; i++)
            for (int j = i + 1; j < n; j++)
                if ((arr[i] > arr[j]) != (arr2[i] > arr2[j]))
                    inv_count++;
    }
    else{
        for (int i = 0; i < n ; i++)
            inv_count=inv_count+abs(arr[i]-arr2[i]);
    }

    return inv_count;
}



int spearDistance_election(int n,int m, const std::vector<std::vector<int>> &
el1, const std::vector<std::vector<int>> & el2){
    int min_dist=2*m*m*n;
    int* mapping = new int [m];
    col *rowsol;
    row *colsol;
    cost *u;
    cost *v;
    rowsol = new col[n];
    colsol = new row[n];
    u = new cost[n];
    v = new cost[n];
    int** costMatrix;
    costMatrix = new int*[n];
    for(int t=0;t<n;t++){
        costMatrix[t]  =  new int[n];
    }
    int** e1mapped_reversed;
    int** e2reversed;
    e1mapped_reversed = new int*[n];
    e2reversed = new int*[n];
    for(int t=0;t<n;t++){
        e1mapped_reversed[t]  =  new int[m];
        e2reversed[t]  =  new int[m];
    }
    for(int i=0; i<m; i++){mapping[i]=i;}
    do {
        for(int t=0; t<n; t++) {
            for (int j = 0; j < m; j++) {
                e1mapped_reversed[t][mapping[el1[t][j]]] = j;
                e2reversed[t][el2[t][j]] = j;

            }
        }
        for(int t=0; t<n; t++){
            for(int j=0; j<n; j++){
                costMatrix[t][j]  =getInvCount(e1mapped_reversed[t],e2reversed[j],m,false);
            }
        }

        int dist = lap(n,costMatrix, rowsol, colsol, u, v);
        if(dist<min_dist){min_dist=dist;}
    } while ( std::next_permutation(mapping,mapping+m) );
    delete[] mapping;
    delete[] rowsol;
    delete[] colsol;
    delete[] u;
    delete[] v;
    for( int i = 0 ; i < n ; i++ )
    {
        delete[] costMatrix[i];
        delete[] e1mapped_reversed[i];
        delete[] e2reversed[i];
    }
    delete[] costMatrix;
    delete[] e1mapped_reversed;
    delete[] e2reversed;
    return min_dist;
}

uint8_t getIvCount(int* arr, int n)
{
    uint8_t inv_count = 0;
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (arr[i] > arr[j])
                inv_count++;
    return inv_count;
}

int swapDistance_election(int n,int m, const std::vector<std::vector<int>> &el1,
const std::vector<std::vector<int>> &el2, uint8_t* lookup){
    int min_dist=2*m*m*n;
    int* mapping = new int [m];
    col *rowsol;
    row *colsol;
    cost *u;
    cost *v;
    rowsol = new col[n];
    colsol = new row[n];
    u = new cost[n];
    v = new cost[n];
    int** costMatrix;
    costMatrix = new int*[n];
    for(int t=0;t<n;t++){
        costMatrix[t]  =  new int[n];
    }
    int** e1mapped_reversed;
    int** e2;
    e1mapped_reversed = new int*[n];
    e2 = new int*[n];
    int votecomb = 0;
    for(int t=0;t<n;t++){
        e1mapped_reversed[t]  =  new int[m];
        e2[t]  =  new int[m];
    }
    for(int i=0; i<m; i++){mapping[i]=i;}
    do {
        for(int t=0; t<n; t++) {
            for (int j = 0; j < m; j++) {
                e1mapped_reversed[t][mapping[el1[t][j]]] = j;
                e2[t][j]=el2[t][j];
            }
        }
//
        for(int t=0; t<n; t++){
            for(int j=0; j<n; j++){
                switch (m) {
                    case 10: votecomb=e1mapped_reversed[t][e2[j][0]]*100000000+e1mapped_reversed[t][e2[j][1]]*10000000+e1mapped_reversed[t][e2[j][2]]*1000000+
                                      e1mapped_reversed[t][e2[j][3]]*100000+e1mapped_reversed[t][e2[j][4]]*10000+e1mapped_reversed[t][e2[j][5]]*1000+e1mapped_reversed[t][e2[j][6]]*100+e1mapped_reversed[t][e2[j][7]]*10+e1mapped_reversed[t][e2[j][8]];break;
                    case 9: votecomb=e1mapped_reversed[t][e2[j][0]]*10000000+e1mapped_reversed[t][e2[j][1]]*1000000+e1mapped_reversed[t][e2[j][2]]*100000+
                                     e1mapped_reversed[t][e2[j][3]]*10000+e1mapped_reversed[t][e2[j][4]]*1000+e1mapped_reversed[t][e2[j][5]]*100+e1mapped_reversed[t][e2[j][6]]*10+e1mapped_reversed[t][e2[j][7]];break;      //execution starts at this case label
                    case 8: votecomb=e1mapped_reversed[t][e2[j][0]]*1000000+e1mapped_reversed[t][e2[j][1]]*100000+e1mapped_reversed[t][e2[j][2]]*10000+
                                     e1mapped_reversed[t][e2[j][3]]*1000+e1mapped_reversed[t][e2[j][4]]*100+e1mapped_reversed[t][e2[j][5]]*10+e1mapped_reversed[t][e2[j][6]];break;
                    case 7: votecomb=e1mapped_reversed[t][e2[j][0]]*100000+e1mapped_reversed[t][e2[j][1]]*10000+e1mapped_reversed[t][e2[j][2]]*1000+
                                    e1mapped_reversed[t][e2[j][3]]*100+e1mapped_reversed[t][e2[j][4]]*10+e1mapped_reversed[t][e2[j][5]];break;
                    case 6: votecomb=e1mapped_reversed[t][e2[j][0]]*10000+e1mapped_reversed[t][e2[j][1]]*1000+e1mapped_reversed[t][e2[j][2]]*100+
                                     e1mapped_reversed[t][e2[j][3]]*10+e1mapped_reversed[t][e2[j][4]];break;
                    case 5: votecomb=e1mapped_reversed[t][e2[j][0]]*1000+e1mapped_reversed[t][e2[j][1]]*100+e1mapped_reversed[t][e2[j][2]]*10+
                                     e1mapped_reversed[t][e2[j][3]];break;
                    case 4: votecomb=e1mapped_reversed[t][e2[j][0]]*100+e1mapped_reversed[t][e2[j][1]]*10+e1mapped_reversed[t][e2[j][2]];break;
                    case 3: votecomb=e1mapped_reversed[t][e2[j][0]]*10+e1mapped_reversed[t][e2[j][1]];break;
                }

                costMatrix[t][j]=lookup[votecomb];
            }
        }

        int dist = lap(n,costMatrix, rowsol, colsol, u, v);
        if(dist<min_dist){min_dist=dist;}
    } while ( std::next_permutation(mapping,mapping+m) );
    delete[] mapping;
    delete[] rowsol;
    delete[] colsol;
    delete[] u;
    delete[] v;
    for( int i = 0 ; i < n ; i++ )
    {
        delete[] costMatrix[i];
        delete[] e1mapped_reversed[i];
        delete[] e2[i];
    }
    delete[] costMatrix;
    delete[] e1mapped_reversed;
    delete[] e2;

    return min_dist;
}

uint8_t* prec_map(int m){
    uint8_t* swap_lookup = new uint8_t[999999999];
    int* mapping = new int [m];
    for(int i=0; i<m; i++){mapping[i]=i;}
    do {
        uint32_t id=0;
        for(int t=0; t<m-1; t++) {
            id=id+mapping[t]*pow(10,m-t-2);
        }
       swap_lookup[id]=getIvCount(mapping, m);
    } while ( std::next_permutation(mapping,mapping+m) );

    delete mapping;
    return swap_lookup;
}

int compute_swap(const std::vector<std::vector<int>>  & elc1, const
std::vector<std::vector<int>> & elc2){
  int mm = elc1[0].size();
  int nn = elc1.size();
  uint8_t* sswap_look=prec_map(mm);
  int ddistance;
  ddistance=swapDistance_election(nn, mm, elc1, elc2, sswap_look);
  delete sswap_look;
  return ddistance;
}

int compute_spear(const std::vector<std::vector<int>>  & elc1, const
std::vector<std::vector<int>> & elc2){
  int mm = elc1[0].size();
  int nn = elc1.size();
  int ddistance;
  ddistance=spearDistance_election(nn, mm,elc1, elc2);
  return ddistance;
}

PYBIND11_MODULE(cppdistances, m) {
    m.doc() = "C++ extension computing the swap and the Spearman distances";
    m.def("swapd", &compute_swap, "Computes the swap distance between two elections.");
    m.def("speard", &compute_spear, "Computes the Spearman distance between two elections.");
}

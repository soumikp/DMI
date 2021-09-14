#include <vector>
#include <map>
#include <algorithm>
#include <RcppArmadillo.h>

//[[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace std;
using namespace arma;
arma::mat tiedrank(arma::mat x){
  int nrow=x.n_rows,ncol=x.n_cols;
  arma::mat x_sorted=sort(x);
  arma::mat t_rank;t_rank.zeros(nrow,ncol);
  arma::mat rank;
  rank.zeros(x.n_rows,x.n_cols);
  for(int i=0;i<ncol;i++){

    map<double, double> val2t_rank;
    double flag=x_sorted(0,i);
    int ppos=0;
    for(int j=1;j<nrow;j++){
      if((abs(x_sorted(j,i)-flag)>1e-12)){
        for(int k=ppos;k<j;k++){
          rank(k,i)=(ppos+j-1.0)/2;
          val2t_rank.insert(make_pair(x_sorted(k,i),(ppos+j-1.0)/2));
        }
        flag=x_sorted(j,i);
        ppos=j;
      }
    }
    int j = x.n_rows;
    for(int k=ppos;k<j;k++){
      rank(k,i)=(ppos+j-1.0)/2;
      val2t_rank.insert(make_pair(x_sorted(k,i),(ppos+j-1.0)/2));
    }

    for(int j=0;j<nrow;j++){
      t_rank(j,i)=val2t_rank[x(j,i)];
    }
  }
  return t_rank;
}

//[[Rcpp::export]]
double jmiCpp(const arma::mat& x,const arma::mat& y, const arma::mat& w){
  int n = x.n_rows, p=x.n_cols;
  int q = y.n_cols;

  arma::mat nx = tiedrank(x)/(n);
  arma::mat ny = tiedrank(y)/(n);

  arma::cube kernelx2;
  kernelx2.zeros(n,n,p);

  for(int ip=0;ip<p;ip++){
    arma::mat kx=repmat(nx.col(ip),1,n);
    kernelx2.slice(ip)=abs(kx-trans(kx));
  }

  arma::cube kernelx21=kernelx2;
  arma::cube kernelx22=pow(kernelx2,2.0);
  arma::cube kernelx23=pow(kernelx2,3.0);
  arma::cube kernelx24=pow(kernelx2,4.0);

  arma::cube kernely2;
  kernely2.zeros(n,n,q);

  for(int iq=0;iq<q;iq++){
    arma::mat ky=repmat(ny.col(iq),1,n);
    kernely2.slice(iq)=abs(ky-trans(ky));
  }

  arma::cube kernely21=kernely2;
  arma::cube kernely22=pow(kernely2,2.0);
  arma::cube kernely23=pow(kernely2,3.0);
  arma::cube kernely24=pow(kernely2,4.0);

  double sigma=sqrt(n/(12*(n+1.0)));
  double h=sigma/(pow(n,1/(p+q+3.0)));
  static const double H[50] = { 0.02, 0.08, 0.18, 0.32, 0.50000000000000011,
                                0.72, 0.98000000000000009, 1.28, 1.6199999999999999, 2.0000000000000004,
                                2.42, 2.88, 3.3800000000000003, 3.9200000000000004, 4.5, 5.12,
                                5.7800000000000011, 6.4799999999999995, 7.22, 8.0000000000000018,
                                8.8199999999999985, 9.68, 10.58, 11.52, 12.5, 13.520000000000001,
                                14.580000000000002, 15.680000000000001, 16.82, 18.0, 19.220000000000002,
                                20.48, 21.78, 23.120000000000005, 24.499999999999996, 25.919999999999998,
                                27.38, 28.88, 30.42, 32.000000000000007, 33.62, 35.279999999999994, 36.98,
                                38.72, 40.5, 42.32, 44.18, 46.08, 48.019999999999996, 50.0};
  int nH=50;
  arma::colvec mi(50);
  arma::colvec Fi(50);
  for(int iw=0;iw<nH;iw++){

    double bw=H[iw];
    double hw=bw*h;

    arma::cube kernelx = 1/(1.0+4*kernelx21/hw+6*kernelx22/(pow(hw,2))+4*kernelx23/(pow(hw,3))+kernelx24/(pow(hw,4)));
    arma::mat kx;
    kx.ones(n,n);
    for (int ip=0; ip<p;ip++){
      kx=kernelx.slice(ip)%kx;
    }
    arma::mat kx1=kx;
    for (int i=0;i<n;i++){
      kx(i,i)=0;
    }
    arma::colvec fx=mean(kx,1);

    arma::cube kernely = 1/(1.0+4.0*kernely21/hw+6.0*kernely22/(pow(hw,2.0))+4.0*kernely23/(pow(hw,3.0))+kernely24/(pow(hw,4.0)));
    arma::mat ky;
    ky.ones(n,n);
    for (int iq=0; iq<q;iq++){
      ky=kernely.slice(iq)%ky;
    }
    arma::mat ky1=ky;
    for (int i=0;i<n;i++){
      ky(i,i)=0;
    }
    arma::colvec fy=mean(ky,1);


    arma::colvec fxfy=fx%fy+1e-100;

    arma::colvec A=log(mean(kx%ky,1)/fxfy);
    mi[iw]=mean(A%w);
  }
  return max(mi);
}


//[[Rcpp::export]]
double entropyCpp(const arma::mat& x, const arma::mat& w){
  int n = x.n_rows;
  arma::mat kx = repmat(x.col(0), 1, n);
  arma::mat kernelx2 = abs(kx - trans(kx));

  arma::mat kernelx21 = kernelx2;
  arma::mat kernelx22 = pow(kernelx2, 2.0);
  arma::mat kernelx23 = pow(kernelx2, 3.0);
  arma::mat kernelx24 = pow(kernelx2, 4.0);

  int dense = pow(n, 1);
  double low = pow(n, -0.50);
  double high = pow(n, -0.10);
  double diff = (high - low)/double(dense - 1);

  NumericVector bw(dense);

  for(int i=0; i<dense; i++){
    if(i == 0){
      bw[i] = low;
    }else if(i == (dense-1)){
      bw[i] = high;
    }else{
      bw[i] = low + diff*i;
    }
  }

  int nH=dense;
  arma::colvec ent(dense);
  for(int iw = 0; iw<nH; iw++){
    double hw = bw[iw];
    arma::mat  kernelx = (1.5/hw)/(1.0 + 4*kernelx21/hw + 6*kernelx22/pow(hw, 2.0) + 4*kernelx23/pow(hw, 3.0) + kernelx24/pow(hw, 4.0));
    for (int i=0;i<n;i++){
      kernelx(i,i)=0;
    }
    arma::colvec fx = sum(kernelx, 1)/double((n-1));
    arma::colvec A = log(fx);
    ent[iw] = -mean(A%w);
  }
  return max(ent);
}


//[[Rcpp::export]]
double indepCpp(const arma::mat& x, const arma::mat& y, const arma::mat& w){
  double mi = jmiCpp(x, y, w);
  double ent_x = entropyCpp(x, w);
  double ent_y = entropyCpp(y, w);
  double ent_total = ent_x + ent_y - mi;
  double cent_x_y = exp(ent_total - ent_y);
  double cent_y_x = exp(ent_total - ent_x);
  return 2*mi*cent_x_y/double(cent_x_y + cent_y_x);
}

//[[Rcpp::export]]
double balanceCpp(const arma::mat& x,const arma::mat& y, const arma::mat& w){
  double mi = jmiCpp(x, y, w);
  double ent_x = entropyCpp(x, w);
  double ent_y = entropyCpp(y, w);
  double ent_total = ent_x + ent_y - mi;
  double cent_x_y = exp(ent_total - ent_y);
  double cent_y_x = exp(ent_total - ent_x);
  return 2*mi*(double(cent_x_y - cent_y_x))/double(cent_x_y + cent_y_x);
}

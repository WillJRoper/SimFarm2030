functions {
    real yield(real[] temp,  
               real mu_t,   
               real sigma_t,  
               real[] precip,  
               real mu_p,   
               real sigma_p, 
               real rho,
               real norm){
        real dy[366];
        int reci;
        for (i in 1:366){
            reci = i;
            if (temp[reci] == -999) dy[reci]=0; 
            else dy[reci]=norm*exp(-(1/(2 - 2*square(rho)))*(  square( (temp[reci]-mu_t)/sigma_t) 
                                                            +  square( (precip[reci]- mu_p)/sigma_p)
                                                            -  2*rho*(temp[reci]-mu_t)*(precip[reci]- mu_p)/(sigma_t*sigma_p)
                                                            ) 
                                  ); 
        }
        return sum(dy);
    }
}

data {
    int<lower=0> n_regions;
    int<lower=0> n_years;
    real d_temp[n_regions,n_years,366];
    real d_precip[n_regions,n_years,366];
    real d_yields[n_regions,n_years];
    int n_gf;
    real temp[n_gf];
    real precip[n_gf];
}



parameters {
    real mu_t;
    real<lower=0.0> sigma_t;
    real mu_p;
    real<lower=0.0> sigma_p;
    real rho;
    real<lower=0.0> norm;
    real<lower=0.0> sig_y;
}

model {

    mu_t ~ normal(0,7);
    sigma_t ~ normal(3,3);
    mu_p ~ normal(0,50);
    sigma_p ~ normal(50,10);
    rho ~ normal(0,1.0);
    norm ~ normal(1,3);
    sig_y ~ normal(3,2);
    for (n in 1:n_regions){
        for (y in 1:n_years){       
            d_yields[n,y]~normal(yield(d_temp[n,y,:],  
                                       mu_t,   
                                       sigma_t,  
                                       d_precip[n,y,:],  
                                       mu_p,  
                                       sigma_p,  
                                       rho,
                                       norm),
                                  sig_y);
        }
    }
}
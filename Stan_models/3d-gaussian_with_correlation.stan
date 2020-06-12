functions {
    real yield(real[] temp,  
               real mu_t,   
               real sigma_t,  
               real[] precip,  
               real mu_p,   
               real sigma_p,
               real[] sun,  
               real mu_s,   
               real sigma_s,
               real rho_tp,
               real rho_ts,
               real rho_ps,
               real norm){
        real dy[6];
        int reci;
        for (i in 1:6){
            reci = i;
            dy[reci]=norm * exp(-(0.5 * 1 / (1 - square(rho_tp) - square(rho_ts) - square(rho_ps) + 2 * rho_tp * rho_ts * rho_ps))
                                *(square((temp[reci]-mu_t)/sigma_t) + square((precip[reci]- mu_p)/sigma_p) + square((sun[reci]- mu_s)/sigma_s)
                                  + 2 * ((temp[reci]-mu_t) * (precip[reci]- mu_p) * (rho_ts*rho_ps - rho_tp) / (sigma_t*sigma_p)
                                       + (temp[reci]-mu_t) * (sun[reci]- mu_s) * (rho_tp*rho_ts - rho_ps) / (sigma_t*sigma_s)
                                       + (precip[reci]-mu_p) * (sun[reci]- mu_s) * (rho_tp*rho_ts - rho_ps) / (sigma_s*sigma_p)
                                       )) 
                              );
        }
        return sum(dy);
    }
}

data {
    int<lower=0> n_regions;
    int<lower=0> n_years;
    real d_temp[n_regions,n_years,6];
    real d_precip[n_regions,n_years,6];
    real d_sun[n_regions,n_years,6];
    real d_yields[n_regions,n_years];
    int n_gf;
    real temp[n_gf];
    real precip[n_gf];
    real sun[n_gf];
}



parameters {
    real mu_t;
    real<lower=0.0> sigma_t;
    real mu_p;
    real<lower=0.0> sigma_p;
    real mu_s;
    real<lower=0.0> sigma_s;
    real rho_tp;
    real rho_ts;
    real rho_ps;
    real<lower=0.0> norm;
}

model {

    mu_t ~ normal(20,5);
    sigma_t ~ normal(5,1);
    mu_p ~ normal(100,25);
    sigma_p ~ normal(25,5);
    mu_s ~ normal(150,25);
    sigma_s ~ normal(25,5);
    rho_tp ~ normal(0.5,0.5);
    rho_ts ~ normal(0.5,0.5);
    rho_ps ~ normal(0.5,0.5);
    norm ~ normal(1,3);
    for (n in 1:n_regions){
        for (y in 1:n_years){       
            d_yields[n,y]~normal(yield(d_temp[n,y,:],  
                                       mu_t,   
                                       sigma_t,  
                                       d_precip[n,y,:],  
                                       mu_p,  
                                       sigma_p,
                                       d_sun[n,y,:],  
                                       mu_s,  
                                       sigma_s,    
                                       rho_tp,
                                       rho_ts,
                                       rho_ps,
                                       norm),
                                  1.0);
        }
    }
}
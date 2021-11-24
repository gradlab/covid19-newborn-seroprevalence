data {
    
    // Antibody data
    int<lower = 0> N; // number of antibody tests in the sample
    int<lower = 0, upper = 1> y[N]; // 1 if positive, 0 if negative
    //vector<lower = 0, upper = 1>[N] male; // 0 if female, 1 if male
    int<lower = 1, upper = 18> age[N]; // 1='Under 5 years', ..., 18='85 years and older'
    int<lower = 0> N_community; // number of communities with data
    int<lower = 1, upper = N_community> community[N]; // communities
    //vector[N_community] x_community; // predictors at the community level
    int<lower = 0> N_months; // number of months of data (n=14)
    int<lower = 0> month[N]; // month of study; 0=Nov 2019 and 13=Dec 2020
    
    // Data on test characteristics
    // Sensitivity is a tunable parameter; no data to estimate
    int<lower = 0> y_spec; // number of true negatives in study
    int<lower = 0> n_spec; // number of negative controls in study
    real<lower=0, upper = 1> sens;
    
    // Poststratification matrix
    int<lower = 0> J; // number of population cells, J = 309 * 18 = 5562
    vector<lower = 0>[J] N_pop; // population sizes for poststratification

}


parameters {
    
    real<lower=0, upper = 1> spec; // specificity of test
    vector[3] b; // intercept, month, and month_squared (fixed effects)
    
    // Random effects
    real<lower = 0> sigma_age;
    real<lower = 0> sigma_community;
    real<lower = 0> sigma_monthcat;   
    vector[N_months] a_monthcat; // varying intercepts for categorical month
    vector[18] a_age; // varying intercepts for age category    
    vector[N_community] a_community; // varying intercepts for community    
    
}


model {
   
    vector[N] p;
    vector[N] p_sample;
    
    // Multilevel model for seroprevalence    
    for (n in 1:N) {
        p[n] = inv_logit(b[1] + b[2] * month[n] + b[3] * month[n] * month[n] + a_age[age[n]] + 
                         a_monthcat[(month[n]+1)] + a_community[community[n]]); 
        p_sample[n] = p[n] * sens + (1 - p[n]) * (1 - spec);
    }; 

    // Model for imperfect test performance
    y ~ bernoulli(p_sample);   
    y_spec ~ binomial(n_spec, spec);
    
    // Likelihood distributions for the random effects
    a_age ~ normal(0, sigma_age);
    a_community ~ normal(0, sigma_community);
    a_monthcat ~ normal(0, sigma_monthcat);    
    
    // Priors
    // For regression coeffs b, priors are taken from rstanarm's default adjusted priors
    // https://cran.r-project.org/web/packages/rstanarm/vignettes/priors.html#default-priors-and-scale-adjustments 
    b[1] ~ normal(0, 10);
    b[2] ~ normal(0, 0.65);
    b[3] ~ normal(0, 0.048);
    
    sigma_age ~ normal(0, 5);
    sigma_monthcat ~ normal(0, 5);
    sigma_community ~ normal(0, 5);
}


// Conduct poststratification
generated quantities {
       
    real p_statewide[N_months];
    vector[J] p_pop; // population prevalence in the J poststratification cells
    vector[J*N_months] p_pop_longitudinal; 
    
    int count;
    int count_all = 1; 
    int i_month;
    
    for (i_month_index in 1:N_months) {
        
        i_month = i_month_index - 1;
        count = 1;
        
        for (i_community in 1:N_community) {
            for (i_age in 1:18) {
                p_pop[count] = inv_logit(b[1] + b[2] * i_month + b[3] * i_month * i_month + a_age[i_age] + a_monthcat[i_month_index] +
                                         a_community[i_community]);
                p_pop_longitudinal[count_all] = p_pop[count];
                count += 1;
                count_all += 1;
            }
        }
        
        p_statewide[i_month_index] = sum(N_pop .* p_pop) / sum(N_pop);
    }
}

################################## SOURCE LIBRARIES ##################################
library(data.table);
library(rmutil);

################################## GLOBAL PATHS ##################################
path_output<-"/home/jesus/Desktop/home/gaa/deepsvm"
################################## GLOBAL VARIABLES ##################################
size<-10000
perc_train<-0.7


################################## FEATURES ##################################
x1<-runif(size,0,1)
x2<-runif(size,0,1)

################################## Lineal problem ##################################

## Free noise target
y<- 5*x1 + 3*x2

## Parameters
mu<-runif(1,mean(y),max(y));
sigma<-runif(1,sd(y),max(y));
sigma_cuad<-runif(1,sd(y),max(y));
alpha<-runif(1,2,10);
beta<-runif(1,2,10);
kappa<-runif(1,0.1,10);
lambda<-runif(1,0.1,10);

## Noise
lap_noise <- rlaplace(size,mu,sigma)
gau_noise <- rnorm(size,mu,sigma_cuad)
beta_noise <- rbeta(size,alpha,beta)
weibull_noise <- rweibull(size,kappa,lambda)

## Disturbed targets
y_lap<- y + lap_noise;
y_gau<- y + gau_noise;
y_beta<- y + beta_noise;
y_weibull<- y + weibull_noise;

## Build datasets
dat_free<-data.table(x1=x1,x2=x2,y=y);
dat_lap<-data.table(x1=x1,x2=x2,y=y_lap);
dat_gau<-data.table(x1=x1,x2=x2,y=y_gau);
dat_beta<-data.table(x1=x1,x2=x2,y=y_beta);
dat_weibull<-data.table(x1=x1,x2=x2,y=y_weibull);

## Split into train and test
ind_train<-sample(1:size,perc_train*size);
ind_test<-setdiff(1:size,ind_train);
train_free<-dat_free[ind_train,]
test_free<-dat_free[ind_test,]
train_lap<-dat_lap[ind_train,]
test_lap<-dat_lap[ind_test,]
train_gau<-dat_gau[ind_train,]
test_gau<-dat_gau[ind_test,]
train_beta<-dat_beta[ind_train,]
test_beta<-dat_beta[ind_test,]
train_weibull<-dat_weibull[ind_train,]
test_weibull<-dat_weibull[ind_test,]

## Save datasets
write.table(train_free,file.path(path_output,"lineal_free_train"),row.names = FALSE,sep = ";");
write.table(test_free,file.path(path_output,"lineal_free_test"),row.names = FALSE,sep = ";");
write.table(train_lap,file.path(path_output,"lineal_lap_train"),row.names = FALSE,sep = ";");
write.table(test_lap,file.path(path_output,"lineal_lap_test"),row.names = FALSE,sep = ";");
write.table(train_gau,file.path(path_output,"lineal_gau_train"),row.names = FALSE,sep = ";");
write.table(test_gau,file.path(path_output,"lineal_gau_test"),row.names = FALSE,sep = ";");
write.table(train_beta,file.path(path_output,"lineal_beta_train"),row.names = FALSE,sep = ";");
write.table(test_beta,file.path(path_output,"lineal_beta_test"),row.names = FALSE,sep = ";");
write.table(train_weibull,file.path(path_output,"lineal_weibull_train"),row.names = FALSE,sep = ";");
write.table(test_weibull,file.path(path_output,"lineal_weibull_test"),row.names = FALSE,sep = ";");


################################## Polynomial problem ##################################
## Free noise target
y<- 5*x1^2 + 3*x2

## Parameters
mu<-runif(1,mean(y),max(y));
sigma<-runif(1,sd(y),max(y));
sigma_cuad<-runif(1,sd(y),max(y));
alpha<-runif(1,2,10);
beta<-runif(1,2,10);
kappa<-runif(1,0.1,10);
lambda<-runif(1,0.1,10);

## Noise
lap_noise <- rlaplace(size,mu,sigma)
gau_noise <- rnorm(size,mu,sigma_cuad)
beta_noise <- rbeta(size,alpha,beta)
weibull_noise <- rweibull(size,kappa,lambda)

## Disturbed targets
y_lap<- y + lap_noise;
y_gau<- y + gau_noise;
y_beta<- y + beta_noise;
y_weibull<- y + weibull_noise;

## Build datasets
dat_free<-data.table(x1=x1,x2=x2,y=y);
dat_lap<-data.table(x1=x1,x2=x2,y=y_lap);
dat_gau<-data.table(x1=x1,x2=x2,y=y_gau);
dat_beta<-data.table(x1=x1,x2=x2,y=y_beta);
dat_weibull<-data.table(x1=x1,x2=x2,y=y_weibull);

## Split into train and test
ind_train<-sample(1:size,perc_train*size);
ind_test<-setdiff(1:size,ind_train);
train_free<-dat_free[ind_train,]
test_free<-dat_free[ind_test,]
train_lap<-dat_lap[ind_train,]
test_lap<-dat_lap[ind_test,]
train_gau<-dat_gau[ind_train,]
test_gau<-dat_gau[ind_test,]
train_beta<-dat_beta[ind_train,]
test_beta<-dat_beta[ind_test,]
train_weibull<-dat_weibull[ind_train,]
test_weibull<-dat_weibull[ind_test,]

## Save datasets
write.table(train_free,file.path(path_output,"polynomial_free_train"),row.names = FALSE,sep = ";");
write.table(test_free,file.path(path_output,"polynomial_free_test"),row.names = FALSE,sep = ";");
write.table(train_lap,file.path(path_output,"polynomial_lap_train"),row.names = FALSE,sep = ";");
write.table(test_lap,file.path(path_output,"polynomial_lap_test"),row.names = FALSE,sep = ";");
write.table(train_gau,file.path(path_output,"polynomial_gau_train"),row.names = FALSE,sep = ";");
write.table(test_gau,file.path(path_output,"polynomial_gau_test"),row.names = FALSE,sep = ";");
write.table(train_beta,file.path(path_output,"polynomial_beta_train"),row.names = FALSE,sep = ";");
write.table(test_beta,file.path(path_output,"polynomial_beta_test"),row.names = FALSE,sep = ";");
write.table(train_weibull,file.path(path_output,"polynomial_weibull_train"),row.names = FALSE,sep = ";");
write.table(test_weibull,file.path(path_output,"polynomial_weibull_test"),row.names = FALSE,sep = ";");

# Execute the unmodified author functions; compare each stage with author RDS.
args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(args[1]); out <- file.path(root,"results")
dir.create(out,recursive=TRUE,showWarnings=FALSE)
suppressPackageStartupMessages({library(xts);library(yaml);library(quantmod);library(R6)
  library(tseries);library(PerformanceAnalytics);library(dplyr);library(lubridate);library(jsonlite)})
setwd(file.path(root,"upstream/trading"))
source("master/rcode/logic/strategy.r")
source("master/rcode/logic/WfHelper.r")
source("master/rcode/logic/helpers/datetime_helper.r")
strat <- Strategy$new("master/rcode/logic/strategy_param.yaml")
wf <- WfHelper$new("master/rcode/logic/WfHelper.yaml")
checks <- list(); tables <- list()
compare <- function(actual,expected,label) {
  stopifnot(identical(dim(actual),dim(expected)))
  if (is.xts(actual)) stopifnot(identical(as.numeric(index(actual)),as.numeric(index(expected))))
  av <- as.matrix(actual); ev <- as.matrix(expected)
  if(is.numeric(av)&&is.numeric(ev)) {
    stopifnot(identical(is.na(av),is.na(ev)))
    err <- max(abs(av-ev),na.rm=TRUE);stopifnot(err<1e-9)
  } else {
    # Author metric frames contain numeric values stored as strings. Compare
    # their numbers, not platform-specific last-digit string formatting.
    an<-suppressWarnings(as.numeric(av));en<-suppressWarnings(as.numeric(ev))
    stopifnot(identical(is.na(an),is.na(en)))
    txt<-is.na(an);stopifnot(identical(as.character(av[txt]),as.character(ev[txt])))
    finite<-is.finite(an)&is.finite(en)
    stopifnot(identical(is.infinite(an),is.infinite(en)))
    err<-if(any(finite))max(abs(an[finite]-en[finite])) else 0
    stopifnot(err<1e-9)
  }
  checks[[length(checks)+1]] <<- list(stage=label,values=length(av),max_abs_error=err)
}
folders<-list.dirs(file.path(root,"data/reference"),recursive=TRUE,full.names=TRUE)
folders<-folders[file.exists(file.path(folders,"02_downsampled.rds"))]
for(folder in folders) {
  name<-paste(basename(dirname(dirname(folder))),basename(folder),sep="-")
  cat("REPRODUCE",name,"\n");flush.console()
  p<-read_yaml(file.path(folder,"params.yaml"))
  price<-readRDS(file.path(folder,"02_downsampled.rds"))
  pre<-strat$precalculate(price)
  compare(pre,readRDS(file.path(folder,"03_precalculated.rds")),paste(name,"EMA"))
  pos<-strat$calculate.position(pre)
  compare(pos,readRDS(file.path(folder,"04_positions.rds")),paste(name,"positions"))
  pnl<-strat$calculate.pnl(pos)
  compare(pnl,readRDS(file.path(folder,"05_pnl.rds")),paste(name,"expert PnL"))
  ix<-wf$calculate.indexes(pnl,p$wf$train_length*24,p$wf$test_length*24)
  compare(ix,readRDS(file.path(folder,"06_wf_indexes.rds")),paste(name,"windows"))
  metrics<-wf$calculate.metrics(pnl,ix,NULL,24)
  compare(metrics,readRDS(file.path(folder,"07_wf_metrics.rds")),paste(name,"window metrics"))
  vals<-metrics %>% filter(train_stat=="sharpe") %>% select(-1:-3) %>% select(starts_with("train"))
  bounds<-metrics %>% filter(train_stat=="sharpe") %>% select(train_start,train_end,test_start,test_end)
  best<-max.col(vals,ties.method="first")
  selected<-data.frame(bounds,selected_value=sapply(seq_len(nrow(vals)),function(i) vals[i,best[i]]),selected_col=colnames(vals)[best])
  compare(selected,readRDS(file.path(folder,"08_wf_best_params.rds")),paste(name,"selected parameters"))
  result<-wf$get.trg.rets(pnl,pos,selected,"train_","test_","sharpe")
  result$rowname<-seq(nrow(result))+as.integer(result$test_start[1])-1
  compare(result,readRDS(file.path(folder,"09_wf_test_evalued.rds")),paste(name,"stitched returns"))
  start<-ceiling_date(first(index(price))+days(p$wf$day_clip_front),"day")
  end<-ceiling_date(last(index(price))-days(p$wf$day_clip_back),"day")
  clipped<-result[paste(start,end,sep="/")];r<-clipped$return
  tables[[length(tables)+1]]<-list(name=name,start=as.character(first(index(r))),end=as.character(last(index(r))),
    rows=NROW(r),annualized_mean=as.numeric(mean(r)*8760),annualized_vol=as.numeric(sd(r)*sqrt(8760)),
    sharpe=as.numeric(sharpe(r,8760)),max_drawdown=as.numeric(maxDrawdown(r)),sortino=as.numeric(sortino_ratio(r,8760)),
    log_return_sum=as.numeric(sum(r)))
  write.csv(data.frame(time=as.numeric(index(price)),close=as.numeric(price)),file.path(out,paste0(name,"-prices.csv")),row.names=FALSE)
  write.csv(data.frame(time=as.numeric(index(result)),coredata(result)),file.path(out,paste0(name,"-returns.csv")),row.names=FALSE)
  write.csv(selected,file.path(out,paste0(name,"-selection.csv")),row.names=FALSE)
  saveRDS(list(pre=pre,pos=pos,pnl=pnl,selected=selected,result=result),file.path(out,paste0(name,".rds")))
  write_json(list(checks=checks,tables=tables),file.path(out,"original-reproduction.json"),pretty=TRUE,auto_unbox=TRUE,digits=16)
}
cat("PASS",length(checks),"stages across",length(folders),"runs\n")

**Georgia School Categorization through Covid**

Data downloaded from <https://gosa.georgia.gov/dashboards-data-report-card/downloadable-data> and aggregated/organized with python and manually. Missing data was filled in with the annual category mean or with zeros as appropriate. Revenue and expenditures data was normalized by the district student count.

json for Georgia cities downloaded from <https://github.com/deldersveld/topojson>

Requirements

python version 3.9.7

pandas version 1.3.4

numpy version 1.20.3

sckikit-learn version 1.2.1

matplotlib version 3.4.3

seaborn version 0.11.2

yellowbrick version 1.5

plotly version 5.13.1

descartes-1.1.0

python-kaleido version 0.2.1

geopandas version 0.12.2

Several revenue values are correlated and may be dropped if modeling is to be done. Cluster labeling simply categorizes like school systems.

![Graphical user interface Description automatically generated](media/b59331d26b41799e4c338071acb5ead0.png)

![Graphical user interface Description automatically generated with low confidence](media/e7a367b83b0dbfcbf1b63f84643a15b9.png)

![A picture containing graphical user interface Description automatically generated](media/92a8fc48041a95bb28bee875ec9574d9.png)

**Overall county cluster labeling**

The elbow was selected at 18 clusters for each school year. 10 could possibly have been chosen but there’s little difference in algorithm performance.

![Chart, line chart Description automatically generated](media/288ccae9a41dad14b4cd501f10b45e54.png)

2018-19

![Chart, line chart Description automatically generated](media/96656b6e207c1cd33a4933352dc749f9.png)

2019-20

![Chart, line chart Description automatically generated](media/a42e07baccb6026122c8394440cfc163.png)

2020-21

**County Systems**

![A picture containing text Description automatically generated](media/6f928bd582f2be210af4edfc71aaabeb.png)

![A picture containing text Description automatically generated](media/902770d2ad4d33e0d802e529bb53b2b1.png)

![A picture containing chart Description automatically generated](media/79b33a58675d46e92f0a834fcb2e4cae.png)

The differences between counties were centered on primarily metro county areas in 2018-19 and decreased in 2019-20 which included the start of the pandemic. In 2020-21 as schools reopened changes increased primarily in southern Georgia.

**City Systems**

![Map Description automatically generated](media/492b9333b282180d92f019b0b47d0417.png)

2018-19

![Map Description automatically generated](media/aba186fe69f246aa1b8c362b14e5c7e9.png)

2019-20

![Map Description automatically generated](media/e6200230de91869d60aba76fbd872f21.png)

2020-21

In 2020-21, city categorizations became more homogenous.

**SAT Combined Score**

The differences between counties were centered on primarily metro county areas in 2018-19 and decreased in 2019-20 which included the start of the pandemic. In 2020-21 as schools reopened changes increased primarily in southern Georgia.

**County Systems**

![Chart Description automatically generated](media/daa868eaef15d60cea113b0a7a68eb76.png)

![Chart Description automatically generated](media/54c794af4375f4e47a77c67937fd3c19.png)

![Chart Description automatically generated](media/54bdd56beab223555a6e387f330a8166.png)

**City Systems**

![Map Description automatically generated](media/b52aa937e5890c70f5144b57d81bc93d.png)

2018-19

![Map Description automatically generated](media/01f15168ee1f9b3e3eacfd2f9d93db2a.png)

2019-20

![Map Description automatically generated](media/7bc6ff07a76c4fdd5155d7f5f8a62f11.png)

After 2018-19, city SAT combined scores became more homogenous – specifically increasing in many systems.

2020-21

**Changes over the years per system**

Overall Bar Charts of the categories used that show changes over the 3 years of data used. These plots may also be viewed in larger form at <https://github.com/tmsuidan/GaEduData2018-21/tree/main/images/bar>

Linear regression equations and r2 values are available here:

<https://github.com/tmsuidan/GaEduData2018-21/tree/main/images/rates>

![](media/353a82f19da259c4c914f9dba73dc807.png)![](media/ac0fa3825c3850ce96f8c0a76cc603ae.png)![](media/906f4160e10099b7e4c98a26136b47ca.png)![](media/ae2218d0e3261105edb40399d29642f8.png)![](media/55e8893e9559f41011dc654cdc29787f.png)![](media/8390aeb38b07035cb6319db5c3d9a89c.png)![](media/6a9cea6e09a734ac4fcc83368469afab.png)![](media/6e16a414064925f4c03b071040284741.png)![](media/3605b9edf3e68e50499fde406739007e.png)![](media/44e145e5401490b6a66a562ced503a84.png)![](media/6dc771273d26fb6211c32ee027050888.png)![](media/8fdabf2b668fb28ca607503426facac4.png)![](media/06b7652ba911be47755feb23002d4054.png)![](media/b3909bf917e596367b75e76e706287f7.png)![](media/1238fabe1147f8571335536ecf35b088.png)![](media/86d7c2444eb73ab6d38164cdf932d73f.png)![](media/8782a244cae5fef7974ca70dc4a22ae4.png)![](media/b42522ee2c08dec894275ee949a46861.png)![](media/736e47034b595b41a4c8fe5ba7379dc1.png)![](media/4864a5f43c7455052093c421e41b9a4f.png)![](media/9a8ee16b5477710396713586957b343d.png)![](media/3f111af3c452c0a8cc373b64b21e14e2.png)![](media/c5718203540a9fe6ba3458683fd151a2.png)![](media/d93d546e32f34b17e99425552c893dbb.png)![](media/6c71fb9d4028c4ca78d9649d3b124021.png)![](media/e9dafbf68a9cf9add17173a9b29f15ac.png)![](media/707d1c668d576378943476a73fb2f7c4.png)![](media/3083d74f486a6b76f7e46dca296dfaeb.png)![](media/ffa0f6a1da5a0eda035b5beb8556ddde.png)![](media/608a8eb9c1ee43c39e464ebdcb4261dc.png)![](media/bb534aede06b8b0945553d525d5d7642.png)![](media/cc713a0981871cf87636511b073b5e38.png)![A picture containing calendar Description automatically generated](media/d3561075e23065519062c8f75f28ce24.png)![](media/fc846f804ca353245b3794f4046a13a9.png)![A picture containing timeline Description automatically generated](media/9fb86e4d02aa8452a8d6de0ec732b2d5.png)![](media/d87406416815ba229ea8f3f4ebadf790.png)![A picture containing chart Description automatically generated](media/e5d3eccf599ffceae6518cc67e30c46f.png)![Chart, timeline, bar chart Description automatically generated](media/e6c8e181c62b7a404de63b4a2eb1f229.png)![Chart Description automatically generated](media/b0b5a992addb6db6d4f9ed5301cc26b2.png)![Chart Description automatically generated with medium confidence](media/288ec5fa3e2d4bd05d63440b45b3ddfd.png)![Chart, bar chart, box and whisker chart Description automatically generated](media/d7e6430e635b52a1bc9024c9e8b2f369.png)![A picture containing treemap chart Description automatically generated](media/2e261683c9c6f616a56bc587d078d4c6.png)![Chart, timeline Description automatically generated](media/6867459df8700f7af7e965b8650157d2.png)![Table Description automatically generated](media/102164cfff0cbc8072bbc6998e4d6356.png)![Table Description automatically generated](media/571ad85f7ff51d7627fe68aa1dc5cadd.png)![Chart Description automatically generated](media/5a09192fe94c29cb253c4c88ce0beca0.png)![Calendar Description automatically generated with medium confidence](media/fd62be3fbef0137a5455def5d5fc7819.png)![A picture containing chart Description automatically generated](media/ef9e1935f98c59f3d268a15fa969898d.png)

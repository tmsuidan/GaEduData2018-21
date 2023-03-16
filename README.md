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

![Diagram Description automatically generated with low confidence](media/ec8d4c4b77a58a33d3d809b9b79d208d.png)

![Diagram Description automatically generated with medium confidence](media/85dc2c2fa5f1973d1e00ffa2127c6d94.png)

![A picture containing diagram Description automatically generated](media/28714d8baeef97cacd2c3629e22adf3b.png)

**Overall county cluster labeling**

The elbow was selected at 17 clusters for each school year. 10 could possibly have been chosen but there’s little difference in algorithm performance.

![Chart, line chart Description automatically generated](media/037c1066efc95a5922f07681ec0a740d.png)

2018-19

![Chart, line chart Description automatically generated](media/1062c18ae19c80413347cccce70ccd3b.png)

2019-20

![Chart, line chart Description automatically generated](media/92adc868915a015ebe2b152d2f0d777e.png)

2020-21

**County Systems**

![A picture containing text Description automatically generated](media/0136e4b0265083d6831f73971d257b8e.png)

![A picture containing text Description automatically generated](media/07ba03283c5c039a16011076c3e594c0.png)

![A picture containing chart Description automatically generated](media/4e7b80b9e90a252ee8efb48eac59a96f.png)

The differences between counties were centered on primarily metro county areas in 2018-19 and decreased in 2019-20 which included the start of the pandemic. In 2020-21 as schools reopened changes increased primarily in southern Georgia.

**City Systems**

**![Map Description automatically generated](media/3cdbac090113f150e558344ae4fdeb3e.png)**

2018-19

![Map Description automatically generated](media/98692e48e3323641c2e95ff85f650cbc.png)

2019-20

![Map Description automatically generated](media/df7134824f97257fd702816bef55d1af.png)

2020-21

In 2020-21, city categorizations became more homogenous.

**SAT Combined Score**

The differences between counties were centered on primarily metro county areas in 2018-19 and decreased in 2019-20 which included the start of the pandemic. In 2020-21 as schools reopened changes increased primarily in southern Georgia.

**County Systems**

**![Chart Description automatically generated](media/369740ec8f7fecf62ba240adb785d8a7.png)**

**![Chart Description automatically generated](media/54c794af4375f4e47a77c67937fd3c19.png)**

**![Chart Description automatically generated](media/54bdd56beab223555a6e387f330a8166.png)**

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

![](media/b2b3dc1341755dd9621712b740e456ef.png)![](media/c544cff24640cfc57caa4bf7a54ec2aa.png)![](media/bc37cc33b5ba7a8540d481fc42c30ebe.png)![](media/d611486a6b40ed1e2f7f847e447de70a.png)![](media/4a2f7484ad2bbd75862b17d623b5083d.png)![](media/5221e3551b9b7c47b08062b6add150e2.png)![](media/7f7628eb4cb360a5528288f885f9f5b8.png)![](media/f70ff4c2956b19d324d5d323f4be724a.png)![](media/dd373ea7bb04aebf6714868749b6d7ee.png)![](media/1972b8b7557f2642dc3196a2236e8775.png)![](media/ed68cc2d3a0f22373ab11bb34182c8fa.png)![](media/a9627ab38d3b01a6377bf2a1e3029dbf.png)![](media/f1146fb0ccfdb6b8817b7ccef415e6e1.png)![](media/731d5ceb2eeb06ce3bb420c632c5f668.png)![](media/66bfccc31c5f76ad4c1144b3c3e960b9.png)![](media/8d031dd0e9e8c7e1e9a6ea6fa2542d2e.png)![](media/701f3341786099af7d7af3e7f0c08475.png)![](media/b07f1b2db426b09c74395a2ec862520c.png)![](media/bd974c24383bccd992a5a75d7f90c350.png)![](media/aa7c5a0b5df9ad070a32bf2bf862455d.png)![](media/29d6b007cb4fd86f1e5900da2048d0cb.png)![](media/7a304481d375438eb4dbaa47963dd69b.png)![](media/aa9d69450745c20672ee907a5a0db210.png)![](media/d85c18c3e76cbf350ce9d7c081d9419c.png)![](media/c6ae8941a2cfec47996ecbb0d565fc67.png)![](media/02ee5f00b0bf346f87c07ec9c1e1e651.png)![](media/5bf7e5122212f2840bac0d32b65e30ef.png)![Chart Description automatically generated](media/b1054d51c05cb34100da21cb1e658240.png)![Chart Description automatically generated](media/a18e3951de893d6d050d1bc0e2342921.png)![A screenshot of a computer Description automatically generated with low confidence](media/ee1943da289163635f9aab30173c7ec9.png)![Chart Description automatically generated with medium confidence](media/3f96ce9f251f3a42444f1b7b4ac9215e.png)![A picture containing building Description automatically generated](media/b16322e534851013ea196e73263ff4f6.png)![Chart Description automatically generated](media/3390da307dc7053581ebf4d6a04f7bfe.png)![Chart, calendar Description automatically generated](media/db43cbee50d3d6f794301420456661a3.png)![Chart Description automatically generated with medium confidence](media/86a5ec0399b899e6193d16bc646dd4ed.png)![Chart Description automatically generated](media/fda850535f047a9e2d8933c2ec0040d2.png)![Chart, treemap chart Description automatically generated](media/9df568b7e3bef9d8acbc6e1ad84a2826.png)![Chart Description automatically generated with medium confidence](media/26155744b9c968ee0943ed9de4524c01.png)![A picture containing timeline Description automatically generated](media/b9f29501dd6298afb4ec609342a7c459.png)![Timeline Description automatically generated](media/97bd91dc567890f6f5f37bee999c99f9.png)![Chart Description automatically generated with medium confidence](media/63938688b670e907e4ee0b7bc0b2507d.png)![A picture containing chart Description automatically generated](media/1b15eedadb21b8511900c2b745931ec2.png)![Chart, timeline, bar chart Description automatically generated](media/efd0a83ec831553347f10be96982d4da.png)![A picture containing chart Description automatically generated](media/34894933d817daa9e541517bfecd8876.png)![Chart, treemap chart Description automatically generated](media/6d110d7a3d158805dcdc585498eac071.png)![Chart, timeline, bar chart, box and whisker chart Description automatically generated](media/a0e62241d7f302f6718861a86c97a163.png)![Chart Description automatically generated](media/b1d3636182e202d3b3b8ec3aa63cf076.png)![Chart, timeline, bar chart Description automatically generated](media/f40ddf7ff832fd7e8af340b18f5958a0.png)

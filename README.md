# POCKET: Pruning Random Convolution Kernels for Time Series Classification
![image](https://github.com/ShaowuChen/POCKET/assets/78587515/7cff47b7-df40-46c8-80b1-d2ef6ed88bca)


Feel free to contact me (shaowu-chen@foxmail.com) if you have questions about the paper. 

# Acknowledgement
- The implementation is based on [ROCKET](https://github.com/angus924/rocket) and [S-ROCKET](https://github.com/salehinejad/srocket)

- Many thanks to Angus Dempster and Hojjat Salehinejad for their kind help.

# Idea
- Pruning random kernels via feature selection in the classifier
- Propose an ADMM-based Algorithm
- Propose an accelerated Algorithm: `POCKET`
  - two stages; introduce relatively invariant penalties
  - Prune up to `60%` kernels
  - `11`$\times$ faster than compared methods
  

# Requirements:
- python (3.6.12)
- Sklearn (0.24.2)

# Code Description 
**Note that we have rename our algorithm in the manuscript; ```P-ROCKET``` or ```PROCKET``` in the code corresponds to  ```POCKET```**
```
  ├── ROCKET-PPV-MAX                  : contains code for ROCKET-PPV-MAX
      ├── reproduce_experiments_ucr.py: Main code 
      ├── ADMM_pruner.py              : Our ADMM-based Algorithm
      ├── PROCKET_pruner.py           : Our POCKET Algorithm
      ├── rocket_functions.py         : Generate kernels
      ├── utils.py                    : Process results
  ├── ROCKET-PPV                      : contains code for ROCKET-PPV
  ├── MiniROCKET                      : contains code for MiniROCKET
  ├── demo.txt                        : write down the name of datasets for implementation here
```

# Dataset:
Find and download UCR 2018 on [UCR2018](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)

(We simply fill Nulls with zeros)

# Demo/Training:
- download and save the dataset archive on the root path
- run the following command

```bash
cd ./ROCKET-PPV-MAX 
python reproduce_experiments_ucr.py -o save_path -n 10 -e 50 
```

# Results

## Pruning ROCKET-PPV-MAX on 85 `bake off' datasets

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky" rowspan="3">Dataset</th>
    <th class="tg-0pky" rowspan="3">PF<br>Acc.<br>(%)</th>
    <th class="tg-0pky" rowspan="3">ResNet<br>Acc.<br>(%)</th>
    <th class="tg-0pky" rowspan="3">ITime<br>Acc.<br>(%)</th>
    <th class="tg-c3ow" colspan="8">ROCKET &amp; Pruning</th>
  </tr>
  <tr>
    <th class="tg-c3ow" colspan="4">Acc. (%)</th>
    <th class="tg-c3ow" colspan="2"># Remaining Kernels</th>
    <th class="tg-c3ow" colspan="2">Overall Pruning Time (s)</th>
  </tr>
  <tr>
    <th class="tg-0pky">Unpruned<br>ROCKET</th>
    <th class="tg-0pky">S-ROCKET</th>
    <th class="tg-fymr">POCKET<br>Stage 1</th>
    <th class="tg-fymr">POCKET<br>Stage 2</th>
    <th class="tg-0pky">S-ROCKET</th>
    <th class="tg-fymr">POCKET</th>
    <th class="tg-0pky">S-ROCKET</th>
    <th class="tg-fymr">POCKET</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-4erg">AVERAGE</td>
    <td class="tg-0pky">81.94</td>
    <td class="tg-0pky">82.49</td>
    <td class="tg-0pky">85.03</td>
    <td class="tg-0pky">85.04±0.51</td>
    <td class="tg-0pky">84.88±0.73</td>
    <td class="tg-0pky">85.17±0.84</td>
    <td class="tg-kaq8">85.05±0.72</td>
    <td class="tg-0pky">3861.64</td>
    <td class="tg-kaq8">3226.35</td>
    <td class="tg-0pky">3433.97</td>
    <td class="tg-kaq8">222.95</td>
  </tr>
  <tr>
    <td class="tg-0pky">Adiac</td>
    <td class="tg-0pky">73.40</td>
    <td class="tg-0pky">82.89</td>
    <td class="tg-0pky">83.63</td>
    <td class="tg-0pky">78.13±0.43</td>
    <td class="tg-0pky">78.13±0.43</td>
    <td class="tg-0pky">80.18±0.54</td>
    <td class="tg-0pky">79.87±0.50</td>
    <td class="tg-0pky">10000.00</td>
    <td class="tg-0pky">5000.00</td>
    <td class="tg-0pky">5551.44</td>
    <td class="tg-0pky">270.69</td>
  </tr>
  <tr>
    <td class="tg-0pky">ArrowHead</td>
    <td class="tg-0pky">87.54</td>
    <td class="tg-0pky">84.46</td>
    <td class="tg-0pky">82.86</td>
    <td class="tg-0pky">81.37±1.03</td>
    <td class="tg-0pky">81.77±1.31</td>
    <td class="tg-0pky">80.86±1.93</td>
    <td class="tg-0pky">81.83±1.59</td>
    <td class="tg-0pky">2447.30</td>
    <td class="tg-0pky">2447.30</td>
    <td class="tg-0pky">398.68</td>
    <td class="tg-0pky">136.33</td>
  </tr>
  <tr>
    <td class="tg-0pky">Beef</td>
    <td class="tg-0pky">72.00</td>
    <td class="tg-0pky">75.33</td>
    <td class="tg-0pky">70.00</td>
    <td class="tg-0pky">82.00±3.71</td>
    <td class="tg-0pky">81.00±2.60</td>
    <td class="tg-0pky">82.67±2.49</td>
    <td class="tg-0pky">83.33±3.65</td>
    <td class="tg-0pky">1945.80</td>
    <td class="tg-0pky">1945.80</td>
    <td class="tg-0pky">411.49</td>
    <td class="tg-0pky">133.90</td>
  </tr>
  <tr>
    <td class="tg-0pky">BeetleFly</td>
    <td class="tg-0pky">87.50</td>
    <td class="tg-0pky">85.00</td>
    <td class="tg-0pky">85.00</td>
    <td class="tg-0pky">90.00±0.00</td>
    <td class="tg-0pky">90.00±0.00</td>
    <td class="tg-0pky">89.50±1.50</td>
    <td class="tg-0pky">90.00±0.00</td>
    <td class="tg-0pky">2107.80</td>
    <td class="tg-0pky">2107.80</td>
    <td class="tg-0pky">200.10</td>
    <td class="tg-0pky">120.52</td>
  </tr>
  <tr>
    <td class="tg-0pky">BirdChicken</td>
    <td class="tg-0pky">86.50</td>
    <td class="tg-0pky">88.50</td>
    <td class="tg-0pky">95.00</td>
    <td class="tg-0pky">90.00±0.00</td>
    <td class="tg-0pky">89.00±3.00</td>
    <td class="tg-0pky">90.00±0.00</td>
    <td class="tg-0pky">90.00±0.00</td>
    <td class="tg-0pky">2430.50</td>
    <td class="tg-0pky">2430.50</td>
    <td class="tg-0pky">199.96</td>
    <td class="tg-0pky">121.81</td>
  </tr>
  <tr>
    <td class="tg-0pky">Car</td>
    <td class="tg-0pky">84.67</td>
    <td class="tg-0pky">92.50</td>
    <td class="tg-0pky">90.00</td>
    <td class="tg-0pky">88.33±1.83</td>
    <td class="tg-0pky">88.33±2.69</td>
    <td class="tg-0pky">92.50±0.83</td>
    <td class="tg-0pky">91.67±1.29</td>
    <td class="tg-0pky">3428.50</td>
    <td class="tg-0pky">3428.50</td>
    <td class="tg-0pky">699.14</td>
    <td class="tg-0pky">127.06</td>
  </tr>
  <tr>
    <td class="tg-0pky">CBF</td>
    <td class="tg-0pky">99.33</td>
    <td class="tg-0pky">99.50</td>
    <td class="tg-0pky">99.89</td>
    <td class="tg-0pky">100.00±0.00</td>
    <td class="tg-0pky">99.96±0.05</td>
    <td class="tg-0pky">99.92±0.07</td>
    <td class="tg-0pky">99.98±0.04</td>
    <td class="tg-0pky">1797.90</td>
    <td class="tg-0pky">1797.90</td>
    <td class="tg-0pky">344.33</td>
    <td class="tg-0pky">137.40</td>
  </tr>
  <tr>
    <td class="tg-0pky">ChlCon</td>
    <td class="tg-0pky">63.39</td>
    <td class="tg-0pky">84.36</td>
    <td class="tg-0pky">87.53</td>
    <td class="tg-0pky">81.50±0.49</td>
    <td class="tg-0pky">79.26±1.46</td>
    <td class="tg-0pky">79.43±0.59</td>
    <td class="tg-0pky">80.71±0.60</td>
    <td class="tg-0pky">4066.90</td>
    <td class="tg-0pky">4066.90</td>
    <td class="tg-0pky">2228.87</td>
    <td class="tg-0pky">162.91</td>
  </tr>
  <tr>
    <td class="tg-0pky">CinCECGTorso</td>
    <td class="tg-0pky">93.43</td>
    <td class="tg-0pky">82.61</td>
    <td class="tg-0pky">85.14</td>
    <td class="tg-0pky">83.61±0.55</td>
    <td class="tg-0pky">82.79±0.74</td>
    <td class="tg-0pky">90.86±2.96</td>
    <td class="tg-0pky">88.23±1.64</td>
    <td class="tg-0pky">2443.60</td>
    <td class="tg-0pky">2443.60</td>
    <td class="tg-0pky">482.42</td>
    <td class="tg-0pky">133.08</td>
  </tr>
  <tr>
    <td class="tg-0pky">Coffee</td>
    <td class="tg-0pky">100.00</td>
    <td class="tg-0pky">100.00</td>
    <td class="tg-0pky">100.00</td>
    <td class="tg-0pky">100.00±0.00</td>
    <td class="tg-0pky">100.00±0.00</td>
    <td class="tg-0pky">100.00±0.00</td>
    <td class="tg-0pky">100.00±0.00</td>
    <td class="tg-0pky">1805.60</td>
    <td class="tg-0pky">1805.60</td>
    <td class="tg-0pky">269.08</td>
    <td class="tg-0pky">124.87</td>
  </tr>
  <tr>
    <td class="tg-0pky">Computers</td>
    <td class="tg-0pky">64.44</td>
    <td class="tg-0pky">81.48</td>
    <td class="tg-0pky">81.20</td>
    <td class="tg-0pky">76.32±0.84</td>
    <td class="tg-0pky">76.80±0.95</td>
    <td class="tg-0pky">74.60±1.82</td>
    <td class="tg-0pky">77.20±0.76</td>
    <td class="tg-0pky">2674.60</td>
    <td class="tg-0pky">2674.60</td>
    <td class="tg-0pky">1809.18</td>
    <td class="tg-0pky">133.98</td>
  </tr>
  <tr>
    <td class="tg-0pky">CricketX</td>
    <td class="tg-0pky">80.21</td>
    <td class="tg-0pky">79.13</td>
    <td class="tg-0pky">86.67</td>
    <td class="tg-0pky">81.92±0.49</td>
    <td class="tg-0pky">82.05±0.62</td>
    <td class="tg-0pky">82.10±0.66</td>
    <td class="tg-0pky">82.18±0.80</td>
    <td class="tg-0pky">7294.70</td>
    <td class="tg-0pky">7294.70</td>
    <td class="tg-0pky">2821.87</td>
    <td class="tg-0pky">167.17</td>
  </tr>
  <tr>
    <td class="tg-0pky">CricketY</td>
    <td class="tg-0pky">79.38</td>
    <td class="tg-0pky">80.33</td>
    <td class="tg-0pky">85.13</td>
    <td class="tg-0pky">85.38±0.60</td>
    <td class="tg-0pky">85.08±0.56</td>
    <td class="tg-0pky">83.87±0.75</td>
    <td class="tg-0pky">84.90±0.71</td>
    <td class="tg-0pky">5750.10</td>
    <td class="tg-0pky">5250.10</td>
    <td class="tg-0pky">2835.55</td>
    <td class="tg-0pky">166.05</td>
  </tr>
  <tr>
    <td class="tg-0pky">CricketZ</td>
    <td class="tg-0pky">80.10</td>
    <td class="tg-0pky">81.15</td>
    <td class="tg-0pky">85.90</td>
    <td class="tg-0pky">85.44±0.64</td>
    <td class="tg-0pky">85.03±0.69</td>
    <td class="tg-0pky">83.87±0.71</td>
    <td class="tg-0pky">85.10±0.71</td>
    <td class="tg-0pky">7040.30</td>
    <td class="tg-0pky">7040.30</td>
    <td class="tg-0pky">2836.70</td>
    <td class="tg-0pky">169.36</td>
  </tr>
  <tr>
    <td class="tg-0pky">DiaSizRed</td>
    <td class="tg-0pky">96.57</td>
    <td class="tg-0pky">30.13</td>
    <td class="tg-0pky">93.14</td>
    <td class="tg-0pky">97.09±0.61</td>
    <td class="tg-0pky">96.50±0.84</td>
    <td class="tg-0pky">95.59±2.37</td>
    <td class="tg-0pky">97.84±0.55</td>
    <td class="tg-0pky">2423.20</td>
    <td class="tg-0pky">2423.20</td>
    <td class="tg-0pky">209.88</td>
    <td class="tg-0pky">124.90</td>
  </tr>
  <tr>
    <td class="tg-0pky">DisPhaOutAG</td>
    <td class="tg-0pky">73.09</td>
    <td class="tg-0pky">71.65</td>
    <td class="tg-0pky">72.66</td>
    <td class="tg-0pky">75.68±0.63</td>
    <td class="tg-0pky">74.89±0.99</td>
    <td class="tg-0pky">74.89±0.68</td>
    <td class="tg-0pky">73.74±0.98</td>
    <td class="tg-0pky">3284.80</td>
    <td class="tg-0pky">3284.80</td>
    <td class="tg-0pky">1899.12</td>
    <td class="tg-0pky">146.55</td>
  </tr>
  <tr>
    <td class="tg-0pky">DisPhaOutCor</td>
    <td class="tg-0pky">79.28</td>
    <td class="tg-0pky">77.10</td>
    <td class="tg-0pky">79.35</td>
    <td class="tg-0pky">76.74±0.88</td>
    <td class="tg-0pky">77.03±1.30</td>
    <td class="tg-0pky">77.54±0.96</td>
    <td class="tg-0pky">76.27±1.61</td>
    <td class="tg-0pky">3271.00</td>
    <td class="tg-0pky">3271.00</td>
    <td class="tg-0pky">2499.27</td>
    <td class="tg-0pky">146.57</td>
  </tr>
  <tr>
    <td class="tg-0pky">DisPhaTW</td>
    <td class="tg-0pky">65.97</td>
    <td class="tg-0pky">66.47</td>
    <td class="tg-0pky">67.63</td>
    <td class="tg-0pky">71.94±0.00</td>
    <td class="tg-0pky">70.86±1.17</td>
    <td class="tg-0pky">70.29±1.82</td>
    <td class="tg-0pky">68.85±2.14</td>
    <td class="tg-0pky">2389.40</td>
    <td class="tg-0pky">2389.40</td>
    <td class="tg-0pky">2013.88</td>
    <td class="tg-0pky">123.28</td>
  </tr>
  <tr>
    <td class="tg-0pky">Earthquakes</td>
    <td class="tg-0pky">75.40</td>
    <td class="tg-0pky">71.15</td>
    <td class="tg-0pky">74.10</td>
    <td class="tg-0pky">74.82±0.00</td>
    <td class="tg-0pky">74.82±0.00</td>
    <td class="tg-0pky">74.96±0.29</td>
    <td class="tg-0pky">74.96±0.54</td>
    <td class="tg-0pky">3263.70</td>
    <td class="tg-0pky">3263.70</td>
    <td class="tg-0pky">2309.86</td>
    <td class="tg-0pky">135.02</td>
  </tr>
  <tr>
    <td class="tg-0pky">ECG200</td>
    <td class="tg-0pky">90.90</td>
    <td class="tg-0pky">87.40</td>
    <td class="tg-0pky">91.00</td>
    <td class="tg-0pky">90.40±0.49</td>
    <td class="tg-0pky">89.90±0.70</td>
    <td class="tg-0pky">91.20±1.17</td>
    <td class="tg-0pky">90.60±0.66</td>
    <td class="tg-0pky">1088.10</td>
    <td class="tg-0pky">1088.10</td>
    <td class="tg-0pky">784.61</td>
    <td class="tg-0pky">135.81</td>
  </tr>
  <tr>
    <td class="tg-0pky">ECG5000</td>
    <td class="tg-0pky">93.65</td>
    <td class="tg-0pky">93.42</td>
    <td class="tg-0pky">94.09</td>
    <td class="tg-0pky">94.75±0.05</td>
    <td class="tg-0pky">94.68±0.09</td>
    <td class="tg-0pky">93.55±0.60</td>
    <td class="tg-0pky">94.78±0.06</td>
    <td class="tg-0pky">3390.60</td>
    <td class="tg-0pky">3390.60</td>
    <td class="tg-0pky">2687.18</td>
    <td class="tg-0pky">173.00</td>
  </tr>
  <tr>
    <td class="tg-0pky">ECGFiveDays</td>
    <td class="tg-0pky">84.92</td>
    <td class="tg-0pky">97.48</td>
    <td class="tg-0pky">100.00</td>
    <td class="tg-0pky">100.00±0.00</td>
    <td class="tg-0pky">100.00±0.00</td>
    <td class="tg-0pky">100.00±0.00</td>
    <td class="tg-0pky">100.00±0.00</td>
    <td class="tg-0pky">2218.50</td>
    <td class="tg-0pky">2218.50</td>
    <td class="tg-0pky">205.96</td>
    <td class="tg-0pky">124.82</td>
  </tr>
  <tr>
    <td class="tg-0pky">ElectricDev</td>
    <td class="tg-0pky">70.60</td>
    <td class="tg-0pky">72.91</td>
    <td class="tg-0pky">72.27</td>
    <td class="tg-0pky">72.81±0.25</td>
    <td class="tg-0pky">72.72±0.39</td>
    <td class="tg-0pky">72.65±0.38</td>
    <td class="tg-0pky">72.63±0.30</td>
    <td class="tg-0pky">4363.10</td>
    <td class="tg-0pky">4363.10</td>
    <td class="tg-0pky">48114.26</td>
    <td class="tg-0pky">654.77</td>
  </tr>
  <tr>
    <td class="tg-0pky">FaceAll</td>
    <td class="tg-0pky">89.38</td>
    <td class="tg-0pky">83.88</td>
    <td class="tg-0pky">80.41</td>
    <td class="tg-0pky">94.68±0.40</td>
    <td class="tg-0pky">94.64±0.32</td>
    <td class="tg-0pky">94.16±0.67</td>
    <td class="tg-0pky">94.14±0.78</td>
    <td class="tg-0pky">4660.60</td>
    <td class="tg-0pky">4660.60</td>
    <td class="tg-0pky">4373.65</td>
    <td class="tg-0pky">196.47</td>
  </tr>
  <tr>
    <td class="tg-0pky">FaceFour</td>
    <td class="tg-0pky">97.39</td>
    <td class="tg-0pky">95.45</td>
    <td class="tg-0pky">96.59</td>
    <td class="tg-0pky">97.61±0.34</td>
    <td class="tg-0pky">97.61±0.34</td>
    <td class="tg-0pky">98.41±0.56</td>
    <td class="tg-0pky">97.84±0.34</td>
    <td class="tg-0pky">1829.90</td>
    <td class="tg-0pky">1829.90</td>
    <td class="tg-0pky">321.47</td>
    <td class="tg-0pky">125.82</td>
  </tr>
  <tr>
    <td class="tg-0pky">FacesUCR</td>
    <td class="tg-0pky">94.59</td>
    <td class="tg-0pky">95.47</td>
    <td class="tg-0pky">97.32</td>
    <td class="tg-0pky">96.20±0.09</td>
    <td class="tg-0pky">96.20±0.08</td>
    <td class="tg-0pky">96.14±0.16</td>
    <td class="tg-0pky">96.34±0.08</td>
    <td class="tg-0pky">4842.70</td>
    <td class="tg-0pky">4842.70</td>
    <td class="tg-0pky">1599.40</td>
    <td class="tg-0pky">183.20</td>
  </tr>
  <tr>
    <td class="tg-0pky">FiftyWords</td>
    <td class="tg-0pky">83.14</td>
    <td class="tg-0pky">73.96</td>
    <td class="tg-0pky">84.18</td>
    <td class="tg-0pky">82.99±0.41</td>
    <td class="tg-0pky">82.99±0.41</td>
    <td class="tg-0pky">82.00±0.47</td>
    <td class="tg-0pky">82.46±0.47</td>
    <td class="tg-0pky">10000.00</td>
    <td class="tg-0pky">5000.00</td>
    <td class="tg-0pky">8068.53</td>
    <td class="tg-0pky">308.62</td>
  </tr>
  <tr>
    <td class="tg-0pky">Fish</td>
    <td class="tg-0pky">93.49</td>
    <td class="tg-0pky">97.94</td>
    <td class="tg-0pky">98.29</td>
    <td class="tg-0pky">97.83±0.62</td>
    <td class="tg-0pky">98.00±0.69</td>
    <td class="tg-0pky">98.40±0.56</td>
    <td class="tg-0pky">98.74±0.50</td>
    <td class="tg-0pky">2257.90</td>
    <td class="tg-0pky">1757.90</td>
    <td class="tg-0pky">849.90</td>
    <td class="tg-0pky">117.25</td>
  </tr>
  <tr>
    <td class="tg-0pky">FordA</td>
    <td class="tg-0pky">85.46</td>
    <td class="tg-0pky">92.05</td>
    <td class="tg-0pky">94.83</td>
    <td class="tg-0pky">94.43±0.28</td>
    <td class="tg-0pky">94.04±0.26</td>
    <td class="tg-0pky">94.61±0.18</td>
    <td class="tg-0pky">94.05±0.57</td>
    <td class="tg-0pky">2812.60</td>
    <td class="tg-0pky">2812.60</td>
    <td class="tg-0pky">13065.10</td>
    <td class="tg-0pky">273.58</td>
  </tr>
  <tr>
    <td class="tg-0pky">FordB</td>
    <td class="tg-0pky">71.49</td>
    <td class="tg-0pky">91.31</td>
    <td class="tg-0pky">93.65</td>
    <td class="tg-0pky">80.43±0.78</td>
    <td class="tg-0pky">79.81±0.39</td>
    <td class="tg-0pky">80.54±0.36</td>
    <td class="tg-0pky">80.52±0.67</td>
    <td class="tg-0pky">3086.00</td>
    <td class="tg-0pky">3086.00</td>
    <td class="tg-0pky">14281.01</td>
    <td class="tg-0pky">264.23</td>
  </tr>
  <tr>
    <td class="tg-0pky">GunPoint</td>
    <td class="tg-0pky">99.73</td>
    <td class="tg-0pky">99.07</td>
    <td class="tg-0pky">100.00</td>
    <td class="tg-0pky">100.00±0.00</td>
    <td class="tg-0pky">100.00±0.00</td>
    <td class="tg-0pky">99.33±0.00</td>
    <td class="tg-0pky">100.00±0.00</td>
    <td class="tg-0pky">1829.80</td>
    <td class="tg-0pky">1829.80</td>
    <td class="tg-0pky">314.20</td>
    <td class="tg-0pky">104.12</td>
  </tr>
  <tr>
    <td class="tg-0pky">Ham</td>
    <td class="tg-0pky">66.00</td>
    <td class="tg-0pky">75.71</td>
    <td class="tg-0pky">71.43</td>
    <td class="tg-0pky">73.43±1.16</td>
    <td class="tg-0pky">71.62±2.29</td>
    <td class="tg-0pky">77.71±1.29</td>
    <td class="tg-0pky">73.52±2.03</td>
    <td class="tg-0pky">2292.30</td>
    <td class="tg-0pky">2292.30</td>
    <td class="tg-0pky">615.16</td>
    <td class="tg-0pky">105.53</td>
  </tr>
  <tr>
    <td class="tg-0pky">HandOutlines</td>
    <td class="tg-0pky">92.14</td>
    <td class="tg-0pky">91.11</td>
    <td class="tg-0pky">95.95</td>
    <td class="tg-0pky">94.11±0.20</td>
    <td class="tg-0pky">94.05±0.30</td>
    <td class="tg-0pky">94.00±0.52</td>
    <td class="tg-0pky">94.03±0.46</td>
    <td class="tg-0pky">2704.70</td>
    <td class="tg-0pky">2704.70</td>
    <td class="tg-0pky">3849.30</td>
    <td class="tg-0pky">152.11</td>
  </tr>
  <tr>
    <td class="tg-0pky">Haptics</td>
    <td class="tg-0pky">44.45</td>
    <td class="tg-0pky">51.88</td>
    <td class="tg-0pky">56.82</td>
    <td class="tg-0pky">52.11±0.36</td>
    <td class="tg-0pky">52.31±0.75</td>
    <td class="tg-0pky">52.50±0.70</td>
    <td class="tg-0pky">52.89±0.91</td>
    <td class="tg-0pky">3901.70</td>
    <td class="tg-0pky">3901.70</td>
    <td class="tg-0pky">742.73</td>
    <td class="tg-0pky">131.02</td>
  </tr>
  <tr>
    <td class="tg-0pky">Herring</td>
    <td class="tg-0pky">57.97</td>
    <td class="tg-0pky">61.88</td>
    <td class="tg-0pky">70.31</td>
    <td class="tg-0pky">69.53±1.05</td>
    <td class="tg-0pky">67.97±1.60</td>
    <td class="tg-0pky">65.16±3.28</td>
    <td class="tg-0pky">61.72±1.88</td>
    <td class="tg-0pky">2863.80</td>
    <td class="tg-0pky">2863.80</td>
    <td class="tg-0pky">407.39</td>
    <td class="tg-0pky">104.51</td>
  </tr>
  <tr>
    <td class="tg-0pky">InlineSkate</td>
    <td class="tg-0pky">54.18</td>
    <td class="tg-0pky">37.31</td>
    <td class="tg-0pky">48.55</td>
    <td class="tg-0pky">45.87±0.65</td>
    <td class="tg-0pky">45.49±1.17</td>
    <td class="tg-0pky">49.45±0.88</td>
    <td class="tg-0pky">48.42±0.79</td>
    <td class="tg-0pky">4943.00</td>
    <td class="tg-0pky">2943.00</td>
    <td class="tg-0pky">543.33</td>
    <td class="tg-0pky">139.91</td>
  </tr>
  <tr>
    <td class="tg-0pky">InsWinSou</td>
    <td class="tg-0pky">61.87</td>
    <td class="tg-0pky">50.65</td>
    <td class="tg-0pky">63.48</td>
    <td class="tg-0pky">65.66±0.21</td>
    <td class="tg-0pky">65.72±0.24</td>
    <td class="tg-0pky">66.12±0.71</td>
    <td class="tg-0pky">66.23±0.43</td>
    <td class="tg-0pky">2763.50</td>
    <td class="tg-0pky">2763.50</td>
    <td class="tg-0pky">1354.89</td>
    <td class="tg-0pky">133.89</td>
  </tr>
  <tr>
    <td class="tg-0pky">ItaPowDem</td>
    <td class="tg-0pky">96.71</td>
    <td class="tg-0pky">96.30</td>
    <td class="tg-0pky">96.79</td>
    <td class="tg-0pky">96.93±0.09</td>
    <td class="tg-0pky">96.82±0.15</td>
    <td class="tg-0pky">96.95±0.19</td>
    <td class="tg-0pky">96.88±0.12</td>
    <td class="tg-0pky">1050.70</td>
    <td class="tg-0pky">1050.70</td>
    <td class="tg-0pky">423.84</td>
    <td class="tg-0pky">107.16</td>
  </tr>
  <tr>
    <td class="tg-0pky">LarKitApp</td>
    <td class="tg-0pky">78.19</td>
    <td class="tg-0pky">89.97</td>
    <td class="tg-0pky">90.67</td>
    <td class="tg-0pky">90.00±0.40</td>
    <td class="tg-0pky">89.28±0.88</td>
    <td class="tg-0pky">89.01±0.50</td>
    <td class="tg-0pky">89.52±0.68</td>
    <td class="tg-0pky">3535.60</td>
    <td class="tg-0pky">3535.60</td>
    <td class="tg-0pky">1613.64</td>
    <td class="tg-0pky">135.65</td>
  </tr>
  <tr>
    <td class="tg-0pky">Lightning2</td>
    <td class="tg-0pky">86.56</td>
    <td class="tg-0pky">77.05</td>
    <td class="tg-0pky">80.33</td>
    <td class="tg-0pky">76.72±0.66</td>
    <td class="tg-0pky">76.72±0.98</td>
    <td class="tg-0pky">78.52±2.59</td>
    <td class="tg-0pky">80.33±2.07</td>
    <td class="tg-0pky">2626.50</td>
    <td class="tg-0pky">2626.50</td>
    <td class="tg-0pky">379.77</td>
    <td class="tg-0pky">104.81</td>
  </tr>
  <tr>
    <td class="tg-0pky">Lightning7</td>
    <td class="tg-0pky">82.19</td>
    <td class="tg-0pky">84.52</td>
    <td class="tg-0pky">80.82</td>
    <td class="tg-0pky">82.19±0.61</td>
    <td class="tg-0pky">82.88±1.10</td>
    <td class="tg-0pky">80.96±1.88</td>
    <td class="tg-0pky">82.47±1.19</td>
    <td class="tg-0pky">3695.90</td>
    <td class="tg-0pky">3695.90</td>
    <td class="tg-0pky">797.78</td>
    <td class="tg-0pky">111.92</td>
  </tr>
  <tr>
    <td class="tg-0pky">Mallat</td>
    <td class="tg-0pky">95.76</td>
    <td class="tg-0pky">97.16</td>
    <td class="tg-0pky">96.29</td>
    <td class="tg-0pky">95.63±0.21</td>
    <td class="tg-0pky">95.63±0.21</td>
    <td class="tg-0pky">92.99±1.09</td>
    <td class="tg-0pky">95.52±0.26</td>
    <td class="tg-0pky">10000.00</td>
    <td class="tg-0pky">5000.00</td>
    <td class="tg-0pky">694.15</td>
    <td class="tg-0pky">120.00</td>
  </tr>
  <tr>
    <td class="tg-0pky">Meat</td>
    <td class="tg-0pky">93.33</td>
    <td class="tg-0pky">96.83</td>
    <td class="tg-0pky">95.00</td>
    <td class="tg-0pky">94.00±2.00</td>
    <td class="tg-0pky">93.83±1.07</td>
    <td class="tg-0pky">94.17±0.83</td>
    <td class="tg-0pky">94.50±0.76</td>
    <td class="tg-0pky">3599.30</td>
    <td class="tg-0pky">3599.30</td>
    <td class="tg-0pky">480.54</td>
    <td class="tg-0pky">104.11</td>
  </tr>
  <tr>
    <td class="tg-0pky">MedicalImages</td>
    <td class="tg-0pky">75.82</td>
    <td class="tg-0pky">77.03</td>
    <td class="tg-0pky">79.87</td>
    <td class="tg-0pky">79.67±0.37</td>
    <td class="tg-0pky">79.54±0.40</td>
    <td class="tg-0pky">79.05±0.69</td>
    <td class="tg-0pky">79.37±0.67</td>
    <td class="tg-0pky">5141.30</td>
    <td class="tg-0pky">4141.30</td>
    <td class="tg-0pky">2278.48</td>
    <td class="tg-0pky">131.79</td>
  </tr>
  <tr>
    <td class="tg-0pky">MidPhaOutAG</td>
    <td class="tg-0pky">56.23</td>
    <td class="tg-0pky">56.88</td>
    <td class="tg-0pky">53.25</td>
    <td class="tg-0pky">58.64±0.71</td>
    <td class="tg-0pky">61.56±1.36</td>
    <td class="tg-0pky">63.51±2.23</td>
    <td class="tg-0pky">56.17±1.75</td>
    <td class="tg-0pky">2795.10</td>
    <td class="tg-0pky">2795.10</td>
    <td class="tg-0pky">1707.16</td>
    <td class="tg-0pky">114.49</td>
  </tr>
  <tr>
    <td class="tg-0pky">MidPhaOutCor</td>
    <td class="tg-0pky">83.64</td>
    <td class="tg-0pky">80.89</td>
    <td class="tg-0pky">83.51</td>
    <td class="tg-0pky">83.47±0.76</td>
    <td class="tg-0pky">83.92±1.01</td>
    <td class="tg-0pky">83.51±1.65</td>
    <td class="tg-0pky">82.65±1.13</td>
    <td class="tg-0pky">3098.90</td>
    <td class="tg-0pky">3098.90</td>
    <td class="tg-0pky">2288.76</td>
    <td class="tg-0pky">118.32</td>
  </tr>
  <tr>
    <td class="tg-0pky">MiddlePTW</td>
    <td class="tg-0pky">52.92</td>
    <td class="tg-0pky">48.44</td>
    <td class="tg-0pky">51.30</td>
    <td class="tg-0pky">55.78±0.74</td>
    <td class="tg-0pky">54.35±0.82</td>
    <td class="tg-0pky">56.88±1.79</td>
    <td class="tg-0pky">54.87±0.88</td>
    <td class="tg-0pky">3797.90</td>
    <td class="tg-0pky">3797.90</td>
    <td class="tg-0pky">2012.32</td>
    <td class="tg-0pky">119.74</td>
  </tr>
  <tr>
    <td class="tg-0pky">MoteStrain</td>
    <td class="tg-0pky">90.24</td>
    <td class="tg-0pky">92.76</td>
    <td class="tg-0pky">90.34</td>
    <td class="tg-0pky">91.41±0.30</td>
    <td class="tg-0pky">91.53±0.44</td>
    <td class="tg-0pky">91.71±0.29</td>
    <td class="tg-0pky">91.38±0.38</td>
    <td class="tg-0pky">1835.30</td>
    <td class="tg-0pky">1835.30</td>
    <td class="tg-0pky">144.22</td>
    <td class="tg-0pky">105.41</td>
  </tr>
  <tr>
    <td class="tg-0pky">NonInvFECGT1</td>
    <td class="tg-0pky">90.66</td>
    <td class="tg-0pky">94.54</td>
    <td class="tg-0pky">96.23</td>
    <td class="tg-0pky">95.25±0.20</td>
    <td class="tg-0pky">95.25±0.20</td>
    <td class="tg-0pky">95.80±0.14</td>
    <td class="tg-0pky">95.61±0.20</td>
    <td class="tg-0pky">10000.00</td>
    <td class="tg-0pky">5000.00</td>
    <td class="tg-0pky">31179.69</td>
    <td class="tg-0pky">596.24</td>
  </tr>
  <tr>
    <td class="tg-0pky">NonInvFECGT2</td>
    <td class="tg-0pky">93.99</td>
    <td class="tg-0pky">94.61</td>
    <td class="tg-0pky">96.74</td>
    <td class="tg-0pky">96.77±0.19</td>
    <td class="tg-0pky">96.77±0.19</td>
    <td class="tg-0pky">96.74±0.26</td>
    <td class="tg-0pky">96.71±0.18</td>
    <td class="tg-0pky">10000.00</td>
    <td class="tg-0pky">5000.00</td>
    <td class="tg-0pky">31973.74</td>
    <td class="tg-0pky">599.28</td>
  </tr>
  <tr>
    <td class="tg-0pky">OliveOil</td>
    <td class="tg-0pky">86.67</td>
    <td class="tg-0pky">83.00</td>
    <td class="tg-0pky">86.67</td>
    <td class="tg-0pky">91.33±1.63</td>
    <td class="tg-0pky">91.67±1.67</td>
    <td class="tg-0pky">93.00±1.00</td>
    <td class="tg-0pky">93.00±1.00</td>
    <td class="tg-0pky">3194.50</td>
    <td class="tg-0pky">3194.50</td>
    <td class="tg-0pky">299.23</td>
    <td class="tg-0pky">243.63</td>
  </tr>
  <tr>
    <td class="tg-0pky">OSULeaf</td>
    <td class="tg-0pky">82.73</td>
    <td class="tg-0pky">97.85</td>
    <td class="tg-0pky">93.39</td>
    <td class="tg-0pky">94.01±0.42</td>
    <td class="tg-0pky">93.88±0.90</td>
    <td class="tg-0pky">94.17±0.91</td>
    <td class="tg-0pky">93.68±0.92</td>
    <td class="tg-0pky">2544.80</td>
    <td class="tg-0pky">2544.80</td>
    <td class="tg-0pky">1003.20</td>
    <td class="tg-0pky">253.27</td>
  </tr>
  <tr>
    <td class="tg-0pky">PhaOutCor</td>
    <td class="tg-0pky">82.35</td>
    <td class="tg-0pky">83.90</td>
    <td class="tg-0pky">85.43</td>
    <td class="tg-0pky">82.96±0.78</td>
    <td class="tg-0pky">82.56±0.74</td>
    <td class="tg-0pky">82.10±0.45</td>
    <td class="tg-0pky">82.73±0.79</td>
    <td class="tg-0pky">3258.50</td>
    <td class="tg-0pky">3258.50</td>
    <td class="tg-0pky">7441.26</td>
    <td class="tg-0pky">308.97</td>
  </tr>
  <tr>
    <td class="tg-0pky">Phoneme</td>
    <td class="tg-0pky">32.01</td>
    <td class="tg-0pky">33.43</td>
    <td class="tg-0pky">33.54</td>
    <td class="tg-0pky">28.05±0.20</td>
    <td class="tg-0pky">29.41±0.37</td>
    <td class="tg-0pky">24.69±0.41</td>
    <td class="tg-0pky">28.33±0.42</td>
    <td class="tg-0pky">4732.60</td>
    <td class="tg-0pky">4732.60</td>
    <td class="tg-0pky">2607.88</td>
    <td class="tg-0pky">389.57</td>
  </tr>
  <tr>
    <td class="tg-0pky">Plane</td>
    <td class="tg-0pky">100.00</td>
    <td class="tg-0pky">100.00</td>
    <td class="tg-0pky">100.00</td>
    <td class="tg-0pky">100.00±0.00</td>
    <td class="tg-0pky">100.00±0.00</td>
    <td class="tg-0pky">100.00±0.00</td>
    <td class="tg-0pky">100.00±0.00</td>
    <td class="tg-0pky">7082.80</td>
    <td class="tg-0pky">5582.80</td>
    <td class="tg-0pky">577.74</td>
    <td class="tg-0pky">251.66</td>
  </tr>
  <tr>
    <td class="tg-0pky">ProPhaOutAG</td>
    <td class="tg-0pky">84.63</td>
    <td class="tg-0pky">85.32</td>
    <td class="tg-0pky">85.37</td>
    <td class="tg-0pky">85.51±0.22</td>
    <td class="tg-0pky">85.66±0.62</td>
    <td class="tg-0pky">85.80±0.67</td>
    <td class="tg-0pky">85.27±0.43</td>
    <td class="tg-0pky">1705.40</td>
    <td class="tg-0pky">1705.40</td>
    <td class="tg-0pky">1752.40</td>
    <td class="tg-0pky">243.38</td>
  </tr>
  <tr>
    <td class="tg-0pky">ProPhaOutCor</td>
    <td class="tg-0pky">87.32</td>
    <td class="tg-0pky">92.13</td>
    <td class="tg-0pky">93.13</td>
    <td class="tg-0pky">90.24±0.60</td>
    <td class="tg-0pky">89.14±0.74</td>
    <td class="tg-0pky">89.73±0.42</td>
    <td class="tg-0pky">90.65±1.10</td>
    <td class="tg-0pky">3019.40</td>
    <td class="tg-0pky">3019.40</td>
    <td class="tg-0pky">2312.24</td>
    <td class="tg-0pky">248.79</td>
  </tr>
  <tr>
    <td class="tg-0pky">ProPhaTW</td>
    <td class="tg-0pky">77.90</td>
    <td class="tg-0pky">78.05</td>
    <td class="tg-0pky">77.56</td>
    <td class="tg-0pky">81.17±0.66</td>
    <td class="tg-0pky">81.07±0.72</td>
    <td class="tg-0pky">80.49±0.79</td>
    <td class="tg-0pky">79.07±1.03</td>
    <td class="tg-0pky">4931.30</td>
    <td class="tg-0pky">3431.30</td>
    <td class="tg-0pky">1998.84</td>
    <td class="tg-0pky">254.29</td>
  </tr>
  <tr>
    <td class="tg-0pky">RefDev</td>
    <td class="tg-0pky">53.23</td>
    <td class="tg-0pky">52.53</td>
    <td class="tg-0pky">50.93</td>
    <td class="tg-0pky">53.49±0.96</td>
    <td class="tg-0pky">53.20±0.87</td>
    <td class="tg-0pky">54.08±0.73</td>
    <td class="tg-0pky">53.41±1.23</td>
    <td class="tg-0pky">3720.50</td>
    <td class="tg-0pky">3720.50</td>
    <td class="tg-0pky">1635.38</td>
    <td class="tg-0pky">338.01</td>
  </tr>
  <tr>
    <td class="tg-0pky">ScreenType</td>
    <td class="tg-0pky">45.52</td>
    <td class="tg-0pky">62.16</td>
    <td class="tg-0pky">57.60</td>
    <td class="tg-0pky">48.56±1.15</td>
    <td class="tg-0pky">49.20±2.09</td>
    <td class="tg-0pky">46.99±1.54</td>
    <td class="tg-0pky">47.92±0.97</td>
    <td class="tg-0pky">3635.10</td>
    <td class="tg-0pky">3635.10</td>
    <td class="tg-0pky">1639.23</td>
    <td class="tg-0pky">345.80</td>
  </tr>
  <tr>
    <td class="tg-0pky">ShapeletSim</td>
    <td class="tg-0pky">77.61</td>
    <td class="tg-0pky">77.94</td>
    <td class="tg-0pky">98.89</td>
    <td class="tg-0pky">100.00±0.00</td>
    <td class="tg-0pky">99.44±0.86</td>
    <td class="tg-0pky">100.00±0.00</td>
    <td class="tg-0pky">100.00±0.00</td>
    <td class="tg-0pky">2084.20</td>
    <td class="tg-0pky">2084.20</td>
    <td class="tg-0pky">159.30</td>
    <td class="tg-0pky">229.72</td>
  </tr>
  <tr>
    <td class="tg-0pky">ShapesAll</td>
    <td class="tg-0pky">88.58</td>
    <td class="tg-0pky">92.13</td>
    <td class="tg-0pky">92.50</td>
    <td class="tg-0pky">90.72±0.22</td>
    <td class="tg-0pky">90.72±0.22</td>
    <td class="tg-0pky">90.18±0.40</td>
    <td class="tg-0pky">90.42±0.30</td>
    <td class="tg-0pky">10000.00</td>
    <td class="tg-0pky">5000.00</td>
    <td class="tg-0pky">10350.97</td>
    <td class="tg-0pky">446.24</td>
  </tr>
  <tr>
    <td class="tg-0pky">SmaKitApp</td>
    <td class="tg-0pky">74.43</td>
    <td class="tg-0pky">78.61</td>
    <td class="tg-0pky">77.87</td>
    <td class="tg-0pky">81.60±0.46</td>
    <td class="tg-0pky">80.96±1.03</td>
    <td class="tg-0pky">82.27±0.95</td>
    <td class="tg-0pky">80.40±0.96</td>
    <td class="tg-0pky">3184.10</td>
    <td class="tg-0pky">3184.10</td>
    <td class="tg-0pky">1638.70</td>
    <td class="tg-0pky">348.32</td>
  </tr>
  <tr>
    <td class="tg-0pky">SonAIBORobS1</td>
    <td class="tg-0pky">84.58</td>
    <td class="tg-0pky">95.81</td>
    <td class="tg-0pky">88.35</td>
    <td class="tg-0pky">92.26±0.20</td>
    <td class="tg-0pky">92.23±0.39</td>
    <td class="tg-0pky">94.26±0.58</td>
    <td class="tg-0pky">93.16±0.32</td>
    <td class="tg-0pky">1821.50</td>
    <td class="tg-0pky">1821.50</td>
    <td class="tg-0pky">156.69</td>
    <td class="tg-0pky">236.41</td>
  </tr>
  <tr>
    <td class="tg-0pky">SonAIBORobS2</td>
    <td class="tg-0pky">89.63</td>
    <td class="tg-0pky">97.78</td>
    <td class="tg-0pky">95.28</td>
    <td class="tg-0pky">91.22±0.27</td>
    <td class="tg-0pky">91.22±0.52</td>
    <td class="tg-0pky">92.13±0.36</td>
    <td class="tg-0pky">92.24±0.55</td>
    <td class="tg-0pky">3025.80</td>
    <td class="tg-0pky">3025.80</td>
    <td class="tg-0pky">181.42</td>
    <td class="tg-0pky">241.38</td>
  </tr>
  <tr>
    <td class="tg-0pky">StarLightC</td>
    <td class="tg-0pky">98.13</td>
    <td class="tg-0pky">97.18</td>
    <td class="tg-0pky">97.92</td>
    <td class="tg-0pky">98.06±0.05</td>
    <td class="tg-0pky">98.06±0.04</td>
    <td class="tg-0pky">98.01±0.13</td>
    <td class="tg-0pky">98.00±0.13</td>
    <td class="tg-0pky">4035.80</td>
    <td class="tg-0pky">2535.80</td>
    <td class="tg-0pky">4544.64</td>
    <td class="tg-0pky">409.38</td>
  </tr>
  <tr>
    <td class="tg-0pky">Strawberry</td>
    <td class="tg-0pky">96.84</td>
    <td class="tg-0pky">98.05</td>
    <td class="tg-0pky">98.38</td>
    <td class="tg-0pky">98.14±0.08</td>
    <td class="tg-0pky">98.03±0.34</td>
    <td class="tg-0pky">97.62±0.48</td>
    <td class="tg-0pky">98.30±0.27</td>
    <td class="tg-0pky">3239.80</td>
    <td class="tg-0pky">3239.80</td>
    <td class="tg-0pky">2399.56</td>
    <td class="tg-0pky">260.15</td>
  </tr>
  <tr>
    <td class="tg-0pky">SwedishLeaf</td>
    <td class="tg-0pky">94.66</td>
    <td class="tg-0pky">95.63</td>
    <td class="tg-0pky">97.12</td>
    <td class="tg-0pky">96.56±0.29</td>
    <td class="tg-0pky">96.32±0.35</td>
    <td class="tg-0pky">96.45±0.34</td>
    <td class="tg-0pky">96.50±0.27</td>
    <td class="tg-0pky">8106.60</td>
    <td class="tg-0pky">7106.60</td>
    <td class="tg-0pky">3604.54</td>
    <td class="tg-0pky">300.89</td>
  </tr>
  <tr>
    <td class="tg-0pky">Symbols</td>
    <td class="tg-0pky">96.16</td>
    <td class="tg-0pky">90.64</td>
    <td class="tg-0pky">98.19</td>
    <td class="tg-0pky">97.43±0.05</td>
    <td class="tg-0pky">97.43±0.05</td>
    <td class="tg-0pky">97.84±0.16</td>
    <td class="tg-0pky">97.46±0.06</td>
    <td class="tg-0pky">10000.00</td>
    <td class="tg-0pky">5000.00</td>
    <td class="tg-0pky">296.87</td>
    <td class="tg-0pky">162.06</td>
  </tr>
  <tr>
    <td class="tg-0pky">SynCon</td>
    <td class="tg-0pky">99.53</td>
    <td class="tg-0pky">99.83</td>
    <td class="tg-0pky">99.67</td>
    <td class="tg-0pky">99.97±0.10</td>
    <td class="tg-0pky">99.73±0.29</td>
    <td class="tg-0pky">99.03±0.10</td>
    <td class="tg-0pky">99.83±0.22</td>
    <td class="tg-0pky">5810.70</td>
    <td class="tg-0pky">5810.70</td>
    <td class="tg-0pky">1495.16</td>
    <td class="tg-0pky">264.37</td>
  </tr>
  <tr>
    <td class="tg-0pky">ToeSeg1</td>
    <td class="tg-0pky">92.46</td>
    <td class="tg-0pky">96.27</td>
    <td class="tg-0pky">96.93</td>
    <td class="tg-0pky">96.80±0.34</td>
    <td class="tg-0pky">96.62±0.34</td>
    <td class="tg-0pky">95.26±1.25</td>
    <td class="tg-0pky">95.88±0.35</td>
    <td class="tg-0pky">1071.80</td>
    <td class="tg-0pky">1071.80</td>
    <td class="tg-0pky">267.47</td>
    <td class="tg-0pky">244.17</td>
  </tr>
  <tr>
    <td class="tg-0pky">ToeSeg2</td>
    <td class="tg-0pky">86.23</td>
    <td class="tg-0pky">90.62</td>
    <td class="tg-0pky">93.85</td>
    <td class="tg-0pky">92.08±0.35</td>
    <td class="tg-0pky">92.54±1.24</td>
    <td class="tg-0pky">93.77±1.16</td>
    <td class="tg-0pky">92.62±0.92</td>
    <td class="tg-0pky">1091.60</td>
    <td class="tg-0pky">1091.60</td>
    <td class="tg-0pky">258.58</td>
    <td class="tg-0pky">243.47</td>
  </tr>
  <tr>
    <td class="tg-0pky">Trace</td>
    <td class="tg-0pky">100.00</td>
    <td class="tg-0pky">100.00</td>
    <td class="tg-0pky">100.00</td>
    <td class="tg-0pky">100.00±0.00</td>
    <td class="tg-0pky">100.00±0.00</td>
    <td class="tg-0pky">100.00±0.00</td>
    <td class="tg-0pky">100.00±0.00</td>
    <td class="tg-0pky">1826.30</td>
    <td class="tg-0pky">1826.30</td>
    <td class="tg-0pky">866.61</td>
    <td class="tg-0pky">237.49</td>
  </tr>
  <tr>
    <td class="tg-0pky">TwoLeadECG</td>
    <td class="tg-0pky">98.86</td>
    <td class="tg-0pky">100.00</td>
    <td class="tg-0pky">99.56</td>
    <td class="tg-0pky">99.91±0.00</td>
    <td class="tg-0pky">99.91±0.00</td>
    <td class="tg-0pky">99.91±0.00</td>
    <td class="tg-0pky">99.91±0.00</td>
    <td class="tg-0pky">1840.70</td>
    <td class="tg-0pky">1840.70</td>
    <td class="tg-0pky">165.23</td>
    <td class="tg-0pky">297.16</td>
  </tr>
  <tr>
    <td class="tg-0pky">TwoPatterns</td>
    <td class="tg-0pky">99.96</td>
    <td class="tg-0pky">99.99</td>
    <td class="tg-0pky">100.00</td>
    <td class="tg-0pky">100.00±0.00</td>
    <td class="tg-0pky">100.00±0.00</td>
    <td class="tg-0pky">100.00±0.00</td>
    <td class="tg-0pky">100.00±0.00</td>
    <td class="tg-0pky">1474.20</td>
    <td class="tg-0pky">1474.20</td>
    <td class="tg-0pky">4908.68</td>
    <td class="tg-0pky">279.18</td>
  </tr>
  <tr>
    <td class="tg-0pky">UWavGesLibA</td>
    <td class="tg-0pky">97.23</td>
    <td class="tg-0pky">85.95</td>
    <td class="tg-0pky">95.45</td>
    <td class="tg-0pky">97.57±0.08</td>
    <td class="tg-0pky">97.36±0.29</td>
    <td class="tg-0pky">97.50±0.17</td>
    <td class="tg-0pky">97.56±0.15</td>
    <td class="tg-0pky">3914.80</td>
    <td class="tg-0pky">3914.80</td>
    <td class="tg-0pky">5160.15</td>
    <td class="tg-0pky">362.54</td>
  </tr>
  <tr>
    <td class="tg-0pky">UWavGesLibX</td>
    <td class="tg-0pky">82.86</td>
    <td class="tg-0pky">78.05</td>
    <td class="tg-0pky">82.47</td>
    <td class="tg-0pky">85.50±0.21</td>
    <td class="tg-0pky">85.28±0.47</td>
    <td class="tg-0pky">85.14±0.42</td>
    <td class="tg-0pky">85.26±0.28</td>
    <td class="tg-0pky">6413.70</td>
    <td class="tg-0pky">3413.70</td>
    <td class="tg-0pky">5167.12</td>
    <td class="tg-0pky">361.90</td>
  </tr>
  <tr>
    <td class="tg-0pky">UWavGesLibY</td>
    <td class="tg-0pky">76.15</td>
    <td class="tg-0pky">67.01</td>
    <td class="tg-0pky">76.88</td>
    <td class="tg-0pky">77.32±0.23</td>
    <td class="tg-0pky">76.99±0.65</td>
    <td class="tg-0pky">76.62±0.51</td>
    <td class="tg-0pky">77.26±0.48</td>
    <td class="tg-0pky">4142.30</td>
    <td class="tg-0pky">2642.30</td>
    <td class="tg-0pky">5208.30</td>
    <td class="tg-0pky">362.11</td>
  </tr>
  <tr>
    <td class="tg-0pky">UWavGesLibZ</td>
    <td class="tg-0pky">76.40</td>
    <td class="tg-0pky">75.01</td>
    <td class="tg-0pky">76.97</td>
    <td class="tg-0pky">79.13±0.20</td>
    <td class="tg-0pky">78.67±0.45</td>
    <td class="tg-0pky">79.15±0.27</td>
    <td class="tg-0pky">78.96±0.47</td>
    <td class="tg-0pky">2373.20</td>
    <td class="tg-0pky">1873.20</td>
    <td class="tg-0pky">5176.96</td>
    <td class="tg-0pky">361.32</td>
  </tr>
  <tr>
    <td class="tg-0pky">Wafer</td>
    <td class="tg-0pky">99.55</td>
    <td class="tg-0pky">99.86</td>
    <td class="tg-0pky">99.87</td>
    <td class="tg-0pky">99.83±0.01</td>
    <td class="tg-0pky">99.78±0.06</td>
    <td class="tg-0pky">99.86±0.04</td>
    <td class="tg-0pky">99.83±0.06</td>
    <td class="tg-0pky">1636.90</td>
    <td class="tg-0pky">1636.90</td>
    <td class="tg-0pky">4069.17</td>
    <td class="tg-0pky">298.65</td>
  </tr>
  <tr>
    <td class="tg-0pky">Wine</td>
    <td class="tg-0pky">56.85</td>
    <td class="tg-0pky">74.44</td>
    <td class="tg-0pky">66.67</td>
    <td class="tg-0pky">80.93±2.99</td>
    <td class="tg-0pky">80.37±3.72</td>
    <td class="tg-0pky">83.15±5.46</td>
    <td class="tg-0pky">82.96±4.96</td>
    <td class="tg-0pky">2496.60</td>
    <td class="tg-0pky">2496.60</td>
    <td class="tg-0pky">364.49</td>
    <td class="tg-0pky">240.98</td>
  </tr>
  <tr>
    <td class="tg-0pky">WordSynonyms</td>
    <td class="tg-0pky">77.87</td>
    <td class="tg-0pky">62.24</td>
    <td class="tg-0pky">75.55</td>
    <td class="tg-0pky">75.33±0.27</td>
    <td class="tg-0pky">75.34±0.27</td>
    <td class="tg-0pky">75.47±0.73</td>
    <td class="tg-0pky">76.00±0.45</td>
    <td class="tg-0pky">9825.40</td>
    <td class="tg-0pky">5325.40</td>
    <td class="tg-0pky">2422.28</td>
    <td class="tg-0pky">317.08</td>
  </tr>
  <tr>
    <td class="tg-0pky">Worms</td>
    <td class="tg-0pky">71.82</td>
    <td class="tg-0pky">79.09</td>
    <td class="tg-0pky">80.52</td>
    <td class="tg-0pky">73.25±1.04</td>
    <td class="tg-0pky">73.25±1.45</td>
    <td class="tg-0pky">73.12±1.65</td>
    <td class="tg-0pky">72.21±1.32</td>
    <td class="tg-0pky">2563.70</td>
    <td class="tg-0pky">2563.70</td>
    <td class="tg-0pky">877.64</td>
    <td class="tg-0pky">328.10</td>
  </tr>
  <tr>
    <td class="tg-0pky">WormsTwoCla</td>
    <td class="tg-0pky">78.44</td>
    <td class="tg-0pky">74.68</td>
    <td class="tg-0pky">79.22</td>
    <td class="tg-0pky">78.96±1.82</td>
    <td class="tg-0pky">79.35±1.69</td>
    <td class="tg-0pky">80.91±1.65</td>
    <td class="tg-0pky">77.79±1.23</td>
    <td class="tg-0pky">3403.80</td>
    <td class="tg-0pky">3403.80</td>
    <td class="tg-0pky">1026.81</td>
    <td class="tg-0pky">324.07</td>
  </tr>
  <tr>
    <td class="tg-0pky">Yoga</td>
    <td class="tg-0pky">87.86</td>
    <td class="tg-0pky">87.02</td>
    <td class="tg-0pky">90.57</td>
    <td class="tg-0pky">91.16±0.36</td>
    <td class="tg-0pky">90.43±0.43</td>
    <td class="tg-0pky">91.38±0.43</td>
    <td class="tg-0pky">91.48±0.52</td>
    <td class="tg-0pky">2140.20</td>
    <td class="tg-0pky">2140.20</td>
    <td class="tg-0pky">1680.05</td>
    <td class="tg-0pky">237.54</td>
  </tr>
</tbody>
</table>


## Pruning ROCKET-PPV-MAX on 43 `extra' datasets
![image](https://github.com/ShaowuChen/POCKET/assets/78587515/32c707ee-330a-45ab-8834-a78d1917a408)


## Robust Principle Component Analysis

Repository for algorithms built with Pytorch for robust PCA, along with various utility functions for dealing with images. 
* https://www.jmlr.org/papers/volume21/18-884/18-884.pdf - Sub gradient solver
* https://arxiv.org/pdf/0912.3599.pdf Robust Principle Component Analysis

Both solvers require Numpy and Pytorch.


## Example
This is a simple demo on video set that has been corrupted


```python
# Torch and Numpy are needed for RPCA
import torch
import numpy as np

# Imports for building that data set and showing results.
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import Image, display
import os
from os import chdir
# Import Local Files with scripts
import sys
from Pursuit import *
from SubGD import *
from Utils import *

import time
# Set Device to CUDA to run GPU
# This notebook is run on CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# call the image folder, the build function should 
image_folder = "MovedObject"
#set the directory
os.chdir()
```

### Build the dataset


```python
# pass the image folder to the build function
# start and end allow for users to only work with subsets of a whole folder of images. Here we only work it the portion that contains motion
data_frame_ = build_data_set(image_folder,
                            start=630,
                            end=820,
                            standardized = True )

# This function takes a sequence of frames and compiles into a gif
return_animated_gif(data_frame_, output_filename='ExampleGif.gif', frame_duration=5, image_size=(120, 160))

# Display the corrupted gif
from IPython.display import Image, display
display(Image(filename='ExampleGif.gif'))
```

    GIF saved successfully.
    


    <IPython.core.display.Image object>


## Call SubGD Solver



```python
# If the rank is unknown, pass a list of guesses and solve iteratively, then  store the results using the code that i give below
# we only run for a brief amount of time in this example.

ranks_ = [2]
sparse_results_ = []
for iteration, rank in enumerate(ranks_):
    start_time = time.time()
    subdd_model = SubGD(data_frame_, device_ = device, rank_ = rank)
    print(f"Rank for problem is: {rank}")
    L,S = subdd_model.fit(iterations = 1200,
                        tolerance = .1,
                        mu = .9, 
                        beta_ = .95,
                        gamma_ = .2, 
                        prints = True,
                        iter_prints = 100)
    # On large datasets, we can have 2 instances of a problem floating around for brief periods on device. Consumes VRAM. If solving iteratively delete the object

    del(subdd_model)

    # I suggest removing torch tensors from device before storing
    # best practice is to always clone the tensor.
    sparse = S.clone().detach().cpu().numpy()
    sparse_results_.append(sparse)
    end_time = time.time()
    duration_seconds = end_time - start_time
    print(f"Total duration: {duration_seconds:.2f} seconds")
```

    Rank for problem is: 2
     error rate: 0.6784954730833966 at iteration 100
     error rate: 0.5962345437310731 at iteration 200
     error rate: 0.4467942757271885 at iteration 300
     error rate: 0.35524286351869344 at iteration 400
     error rate: 0.2931604221703815 at iteration 500
     error rate: 0.25008451743468385 at iteration 600
     error rate: 0.22627917449117355 at iteration 700
     error rate: 0.21020006573346095 at iteration 800
     error rate: 0.17579457504463986 at iteration 900
     error rate: 0.1738911433446629 at iteration 1000
     error rate: 0.17368023565056204 at iteration 1100
    Total duration: 63.37 seconds
    

### Returns 


```python
# Extract from the list this way. If on GPU, use .detach().cpu().numpy(). Or just call sparse from prev cell
sparse_show = torch.tensor(sparse_results_[0]).numpy()

# Takes the top 10% of values and maximizes. Rest are 0. 
percentile_animated_gif(sparse_show, output_filename='ExampleRPCA.gif',percentile_cutoff=95, frame_duration=5, image_size=(120, 160))

from IPython.display import Image, display
display(Image(filename='ExampleRPCA.gif'))
```

    GIF saved successfully as ExampleRPCA.gif.
    


    <IPython.core.display.Image object>



```python
# we call a different data set building function that will also tamper with the images. 
data_frame_vc = build_corrupted_imageset(image_folder, start = 600, end=820,method = 'Remove', corrupt_param= .7, standardized=True )

# This function takes a sequence of frames and compiles into a gif
return_animated_gif(data_frame_, output_filename='ExampleGifcorrupted.gif', frame_duration=5, image_size=(120, 160))

# we display the corrupted gif
from IPython.display import Image, display
display(Image(filename='ExampleGifcorrupted.gif'))
```

    GIF saved successfully.
    


    <IPython.core.display.Image object>



```python
# If the rank is unknown, pass a list of guesses and solve iteratively, and store the results using the code that i gave below
# we only run for a brief amount of time on local in this example.
ranks_ = [4]
sparse_results_ = []
for iteration, rank in enumerate(ranks_):
    start_time = time.time()
    subdd_model = SubGD(data_frame_, device_ = device, rank_ = rank)
    print(f"Rank for problem is: {rank}")
    L,S = subdd_model.fit(iterations = 5000,
                        tolerance = .1,
                        mu = .9, 
                        beta_ = .95,
                        gamma_ = .5, 
                        prints = True,
                        iter_prints = 10)
    del(subdd_model) # On large datasets, we can have 2 instances of a problem floating around for brief periods on device. Consumes VRAM.
    # I suggest removing torch tensors from device before storing
    sparsevc = S.clone().detach().cpu().numpy()
    sparse_results_.append(sparse)
    end_time = time.time()
    duration_seconds = end_time - start_time
    print(f"Total duration: {duration_seconds:.2f} seconds")
```

    Rank for problem is: 4
     error rate: 0.7242109180848748 at iteration 10
     error rate: 0.7302469389658374 at iteration 20
     error rate: 0.7321862127089444 at iteration 30
     error rate: 0.7324134826039408 at iteration 40
     error rate: 0.7329908396460513 at iteration 50
     error rate: 0.7339894677646328 at iteration 60
     error rate: 0.7341436921586643 at iteration 70
     error rate: 0.735572539821946 at iteration 80
     error rate: 0.7352008439546858 at iteration 90
     error rate: 0.7375699623856382 at iteration 100
     error rate: 0.7387074805291896 at iteration 110
     error rate: 0.7400675103901952 at iteration 120
     error rate: 0.7391563145135169 at iteration 130
     error rate: 0.7406848594549609 at iteration 140
     error rate: 0.7402641249301632 at iteration 150
     error rate: 0.7412351330768137 at iteration 160
     error rate: 0.7407602238193115 at iteration 170
     error rate: 0.7417087065735457 at iteration 180
     error rate: 0.7420151638077144 at iteration 190
     error rate: 0.7429823924294504 at iteration 200
     error rate: 0.7432421945742643 at iteration 210
     error rate: 0.744169118726518 at iteration 220
     error rate: 0.7445445007237431 at iteration 230
     error rate: 0.7448730519532626 at iteration 240
     error rate: 0.7450606294948632 at iteration 250
     error rate: 0.7457522177461074 at iteration 260
     error rate: 0.745461676855555 at iteration 270
     error rate: 0.7458711439483853 at iteration 280
     error rate: 0.7460868636420833 at iteration 290
     error rate: 0.746196964948733 at iteration 300
     error rate: 0.7460849109954901 at iteration 310
     error rate: 0.7467596338131895 at iteration 320
     error rate: 0.7468212659194107 at iteration 330
     error rate: 0.7471303992599568 at iteration 340
     error rate: 0.7472552087369078 at iteration 350
     error rate: 0.7474397920863246 at iteration 360
     error rate: 0.7485719004319846 at iteration 370
     error rate: 0.7487133885319934 at iteration 380
     error rate: 0.7486363363744419 at iteration 390
     error rate: 0.7502981548525305 at iteration 400
     error rate: 0.7498005526813369 at iteration 410
     error rate: 0.7502056384935415 at iteration 420
     error rate: 0.7495973409368379 at iteration 430
     error rate: 0.7502716398419497 at iteration 440
     error rate: 0.7504130277253271 at iteration 450
     error rate: 0.7505524463880023 at iteration 460
     error rate: 0.7508567841714029 at iteration 470
     error rate: 0.7512967105805756 at iteration 480
     error rate: 0.7511823003919363 at iteration 490
     error rate: 0.7520047103603321 at iteration 500
     error rate: 0.7516142430725964 at iteration 510
     error rate: 0.7524507224883635 at iteration 520
     error rate: 0.7528029357566124 at iteration 530
     error rate: 0.7530938758927945 at iteration 540
     error rate: 0.753000986483011 at iteration 550
     error rate: 0.7530409423498505 at iteration 560
     error rate: 0.7528314789000998 at iteration 570
     error rate: 0.7531310538541145 at iteration 580
     error rate: 0.7525663218745023 at iteration 590
     error rate: 0.7529381920139973 at iteration 600
     error rate: 0.7527215137006011 at iteration 610
     error rate: 0.7527131875768631 at iteration 620
     error rate: 0.7526235811811225 at iteration 630
     error rate: 0.7525681556048358 at iteration 640
     error rate: 0.7525455818336316 at iteration 650
     error rate: 0.752468100707866 at iteration 660
     error rate: 0.7524996233509339 at iteration 670
     error rate: 0.7524604791266163 at iteration 680
     error rate: 0.7525258736242085 at iteration 690
     error rate: 0.7524713494678379 at iteration 700
     error rate: 0.7522535896285732 at iteration 710
     error rate: 0.7525201567333231 at iteration 720
     error rate: 0.7521868479578262 at iteration 730
     error rate: 0.7524736557729154 at iteration 740
     error rate: 0.7524007183185414 at iteration 750
     error rate: 0.7523177987168971 at iteration 760
     error rate: 0.7522833247811799 at iteration 770
     error rate: 0.7520859578613506 at iteration 780
     error rate: 0.7521398495067582 at iteration 790
     error rate: 0.7518174374394332 at iteration 800
     error rate: 0.7519642878727626 at iteration 810
     error rate: 0.7520746947668103 at iteration 820
     error rate: 0.7519251811071428 at iteration 830
     error rate: 0.7518800063863498 at iteration 840
     error rate: 0.7517709829226789 at iteration 850
     error rate: 0.7519351335625898 at iteration 860
     error rate: 0.7517891877759453 at iteration 870
     error rate: 0.7518202207851511 at iteration 880
     error rate: 0.7519186316942293 at iteration 890
     error rate: 0.7519275719113488 at iteration 900
     error rate: 0.7518584209836204 at iteration 910
     error rate: 0.7519288928234765 at iteration 920
     error rate: 0.7521343499281615 at iteration 930
     error rate: 0.7522180092043118 at iteration 940
     error rate: 0.7521446300866823 at iteration 950
     error rate: 0.751996767402965 at iteration 960
     error rate: 0.7521349373205589 at iteration 970
     error rate: 0.752003332403191 at iteration 980
     error rate: 0.7520120610162088 at iteration 990
     error rate: 0.7521630142689669 at iteration 1000
     error rate: 0.7520455358203144 at iteration 1010
     error rate: 0.7522913773605852 at iteration 1020
     error rate: 0.7521416947413447 at iteration 1030
     error rate: 0.7521413183225685 at iteration 1040
     error rate: 0.7520303661712888 at iteration 1050
     error rate: 0.7522216700322552 at iteration 1060
     error rate: 0.7518905402860683 at iteration 1070
     error rate: 0.7521084268563177 at iteration 1080
     error rate: 0.7522005542576066 at iteration 1090
     error rate: 0.7524847934816165 at iteration 1100
     error rate: 0.7522757584430734 at iteration 1110
     error rate: 0.7523045232399114 at iteration 1120
     error rate: 0.7520974475739729 at iteration 1130
     error rate: 0.7523140421882568 at iteration 1140
     error rate: 0.7522499844832292 at iteration 1150
     error rate: 0.7523524904983735 at iteration 1160
     error rate: 0.7522738129659743 at iteration 1170
     error rate: 0.7523649607793766 at iteration 1180
     error rate: 0.7521963214364188 at iteration 1190
     error rate: 0.752337810498209 at iteration 1200
     error rate: 0.7520685495939128 at iteration 1210
     error rate: 0.7522755071135399 at iteration 1220
     error rate: 0.7521804907966682 at iteration 1230
     error rate: 0.7522594252442817 at iteration 1240
     error rate: 0.7519966317977527 at iteration 1250
     error rate: 0.7521180763342117 at iteration 1260
     error rate: 0.7520882626414472 at iteration 1270
     error rate: 0.7521423425587841 at iteration 1280
     error rate: 0.7521348370290237 at iteration 1290
     error rate: 0.7519963383882005 at iteration 1300
     error rate: 0.7519667903361513 at iteration 1310
     error rate: 0.7520253580052949 at iteration 1320
     error rate: 0.7517539459364236 at iteration 1330
     error rate: 0.7520121211515137 at iteration 1340
     error rate: 0.7521281670995261 at iteration 1350
     error rate: 0.7517640452592637 at iteration 1360
     error rate: 0.7519325945670977 at iteration 1370
     error rate: 0.7518821304569234 at iteration 1380
     error rate: 0.7518862039555909 at iteration 1390
     error rate: 0.7523188448654163 at iteration 1400
     error rate: 0.7522867739249006 at iteration 1410
     error rate: 0.7520214114241885 at iteration 1420
     error rate: 0.7522225410623076 at iteration 1430
     error rate: 0.7524374466051661 at iteration 1440
     error rate: 0.7520750142291949 at iteration 1450
     error rate: 0.7522567131252919 at iteration 1460
     error rate: 0.752089525662029 at iteration 1470
     error rate: 0.7522034052954515 at iteration 1480
     error rate: 0.7521056706843335 at iteration 1490
     error rate: 0.752161324486461 at iteration 1500
     error rate: 0.7520852901329019 at iteration 1510
     error rate: 0.752083787275362 at iteration 1520
     error rate: 0.7519115029737584 at iteration 1530
     error rate: 0.7519902278470504 at iteration 1540
     error rate: 0.7519925970969216 at iteration 1550
     error rate: 0.7521861092920739 at iteration 1560
     error rate: 0.7518037679439394 at iteration 1570
     error rate: 0.7519023674101781 at iteration 1580
     error rate: 0.7518418101173523 at iteration 1590
     error rate: 0.7518427785991322 at iteration 1600
     error rate: 0.7517646238192984 at iteration 1610
     error rate: 0.7518497686699783 at iteration 1620
     error rate: 0.7516981221103969 at iteration 1630
     error rate: 0.7519571775290704 at iteration 1640
     error rate: 0.7518053360125622 at iteration 1650
     error rate: 0.751733823257822 at iteration 1660
     error rate: 0.7518496273478366 at iteration 1670
     error rate: 0.7516045315337407 at iteration 1680
     error rate: 0.7516079874712691 at iteration 1690
     error rate: 0.7515720497792792 at iteration 1700
     error rate: 0.7517343044765809 at iteration 1710
     error rate: 0.7515369885163764 at iteration 1720
     error rate: 0.751653335864873 at iteration 1730
     error rate: 0.7515250024901482 at iteration 1740
     error rate: 0.7515786788395727 at iteration 1750
     error rate: 0.7515568430555916 at iteration 1760
     error rate: 0.7515731197949628 at iteration 1770
     error rate: 0.7515774120005001 at iteration 1780
     error rate: 0.7514987148245873 at iteration 1790
     error rate: 0.75143106873153 at iteration 1800
     error rate: 0.7513657364893409 at iteration 1810
     error rate: 0.7514250044628997 at iteration 1820
     error rate: 0.7512430321259961 at iteration 1830
     error rate: 0.7512752913900453 at iteration 1840
     error rate: 0.7511930248260795 at iteration 1850
     error rate: 0.7512051756869627 at iteration 1860
     error rate: 0.7511603364164465 at iteration 1870
     error rate: 0.7510748332104016 at iteration 1880
     error rate: 0.7511259893885657 at iteration 1890
     error rate: 0.7511982752955998 at iteration 1900
     error rate: 0.7511560332219867 at iteration 1910
     error rate: 0.7512081219348037 at iteration 1920
     error rate: 0.751053867380795 at iteration 1930
     error rate: 0.7511362551032856 at iteration 1940
     error rate: 0.7511003618863399 at iteration 1950
     error rate: 0.7512698842389773 at iteration 1960
     error rate: 0.7513094834331653 at iteration 1970
     error rate: 0.7514626944284504 at iteration 1980
     error rate: 0.7513509365308364 at iteration 1990
     error rate: 0.7513580237478245 at iteration 2000
     error rate: 0.7512670056831791 at iteration 2010
     error rate: 0.7513155687933976 at iteration 2020
     error rate: 0.7511979852590411 at iteration 2030
     error rate: 0.7513224882127463 at iteration 2040
     error rate: 0.7511442721939915 at iteration 2050
     error rate: 0.7510667008093247 at iteration 2060
     error rate: 0.7510336039125799 at iteration 2070
     error rate: 0.7509376835113127 at iteration 2080
     error rate: 0.7508808931200125 at iteration 2090
     error rate: 0.7508082132193165 at iteration 2100
     error rate: 0.7507046785932822 at iteration 2110
     error rate: 0.7506821361275623 at iteration 2120
     error rate: 0.7505280643665915 at iteration 2130
     error rate: 0.7505226501247912 at iteration 2140
     error rate: 0.7505600952753803 at iteration 2150
     error rate: 0.7506003203041951 at iteration 2160
     error rate: 0.7505632507122475 at iteration 2170
     error rate: 0.7505220650367278 at iteration 2180
     error rate: 0.7504562016448525 at iteration 2190
     error rate: 0.7504065979608714 at iteration 2200
     error rate: 0.7503299775897437 at iteration 2210
     error rate: 0.7503355601553042 at iteration 2220
     error rate: 0.7501663504056478 at iteration 2230
     error rate: 0.7502495682058984 at iteration 2240
     error rate: 0.7502353064812654 at iteration 2250
     error rate: 0.7501366140002964 at iteration 2260
     error rate: 0.7500908861053529 at iteration 2270
     error rate: 0.7500831935301282 at iteration 2280
     error rate: 0.7500326892752045 at iteration 2290
     error rate: 0.7499132669590495 at iteration 2300
     error rate: 0.7498965133712496 at iteration 2310
     error rate: 0.7498427092553516 at iteration 2320
     error rate: 0.7497856772224154 at iteration 2330
     error rate: 0.7496737145581287 at iteration 2340
     error rate: 0.7496271971318144 at iteration 2350
     error rate: 0.7495594133755207 at iteration 2360
     error rate: 0.7495747475551822 at iteration 2370
     error rate: 0.7494757170131457 at iteration 2380
     error rate: 0.7493891536667134 at iteration 2390
     error rate: 0.7493634814006521 at iteration 2400
     error rate: 0.7492727410862234 at iteration 2410
     error rate: 0.7492315608789578 at iteration 2420
     error rate: 0.7491792242836101 at iteration 2430
     error rate: 0.7491050586352795 at iteration 2440
     error rate: 0.7491019713221208 at iteration 2450
     error rate: 0.7490346640538219 at iteration 2460
     error rate: 0.7489871230261591 at iteration 2470
     error rate: 0.7490143538009193 at iteration 2480
     error rate: 0.7489393549658311 at iteration 2490
     error rate: 0.748867545891649 at iteration 2500
     error rate: 0.7487685715477056 at iteration 2510
     error rate: 0.7487375157223068 at iteration 2520
     error rate: 0.748678579602023 at iteration 2530
     error rate: 0.7485933487407094 at iteration 2540
     error rate: 0.7485613098088203 at iteration 2550
     error rate: 0.7484772601551472 at iteration 2560
     error rate: 0.7484088215248346 at iteration 2570
     error rate: 0.7483091823285842 at iteration 2580
     error rate: 0.7483118123255342 at iteration 2590
     error rate: 0.748203629539845 at iteration 2600
     error rate: 0.7482160505481239 at iteration 2610
     error rate: 0.748120721350967 at iteration 2620
     error rate: 0.7481173905471136 at iteration 2630
     error rate: 0.7480246929494424 at iteration 2640
     error rate: 0.7480211915760147 at iteration 2650
     error rate: 0.7481006383035068 at iteration 2660
     error rate: 0.7479811203291121 at iteration 2670
     error rate: 0.7478906002377246 at iteration 2680
     error rate: 0.7477717186333926 at iteration 2690
     error rate: 0.7477024709055191 at iteration 2700
     error rate: 0.747571288851405 at iteration 2710
     error rate: 0.7474723773025393 at iteration 2720
     error rate: 0.747371917269407 at iteration 2730
     error rate: 0.7472892089242821 at iteration 2740
     error rate: 0.747201032531588 at iteration 2750
     error rate: 0.7471027721420519 at iteration 2760
     error rate: 0.747006564060892 at iteration 2770
     error rate: 0.7469459763174545 at iteration 2780
     error rate: 0.7468030019147927 at iteration 2790
     error rate: 0.7467695226047478 at iteration 2800
     error rate: 0.7466098942387982 at iteration 2810
     error rate: 0.746516657380834 at iteration 2820
     error rate: 0.7464033058320775 at iteration 2830
     error rate: 0.7463239325447277 at iteration 2840
     error rate: 0.7462141758601916 at iteration 2850
     error rate: 0.7461847307942598 at iteration 2860
     error rate: 0.7460343239462688 at iteration 2870
     error rate: 0.746069531031391 at iteration 2880
     error rate: 0.7458982523591965 at iteration 2890
     error rate: 0.7459180060449782 at iteration 2900
     error rate: 0.7457533556749986 at iteration 2910
     error rate: 0.745732821514274 at iteration 2920
     error rate: 0.7455449678447236 at iteration 2930
     error rate: 0.7455393292610119 at iteration 2940
     error rate: 0.7453596746644131 at iteration 2950
     error rate: 0.745392932236125 at iteration 2960
     error rate: 0.7451589415274151 at iteration 2970
     error rate: 0.7451594074762273 at iteration 2980
     error rate: 0.7449266996423959 at iteration 2990
     error rate: 0.744954711326343 at iteration 3000
     error rate: 0.7447069416477569 at iteration 3010
     error rate: 0.7447596070925563 at iteration 3020
     error rate: 0.7445179952828673 at iteration 3030
     error rate: 0.7445289255868387 at iteration 3040
     error rate: 0.7443739072283153 at iteration 3050
     error rate: 0.7443489216798777 at iteration 3060
     error rate: 0.7441990157606254 at iteration 3070
     error rate: 0.7441616104495771 at iteration 3080
     error rate: 0.744016490002345 at iteration 3090
     error rate: 0.7439407870877327 at iteration 3100
     error rate: 0.743794566514764 at iteration 3110
     error rate: 0.743726157283207 at iteration 3120
     error rate: 0.7435784917023712 at iteration 3130
     error rate: 0.743519796224709 at iteration 3140
     error rate: 0.7434040697904432 at iteration 3150
     error rate: 0.7433035662331516 at iteration 3160
     error rate: 0.7431920669761556 at iteration 3170
     error rate: 0.7430828994432388 at iteration 3180
     error rate: 0.7429628357946049 at iteration 3190
     error rate: 0.7428561882615143 at iteration 3200
     error rate: 0.7427311321151021 at iteration 3210
     error rate: 0.7426193879176474 at iteration 3220
     error rate: 0.7425414912729381 at iteration 3230
     error rate: 0.7424224873139542 at iteration 3240
     error rate: 0.7423595396108765 at iteration 3250
     error rate: 0.7421943617711604 at iteration 3260
     error rate: 0.742123040841072 at iteration 3270
     error rate: 0.7420028597492073 at iteration 3280
     error rate: 0.741935743640046 at iteration 3290
     error rate: 0.7418079709200258 at iteration 3300
     error rate: 0.7416976086811204 at iteration 3310
     error rate: 0.7417328927914992 at iteration 3320
     error rate: 0.7416297391745121 at iteration 3330
     error rate: 0.7414733194934832 at iteration 3340
     error rate: 0.7413800001511462 at iteration 3350
     error rate: 0.7412547067146696 at iteration 3360
     error rate: 0.7411478508769201 at iteration 3370
     error rate: 0.741025369107083 at iteration 3380
     error rate: 0.7409515826113485 at iteration 3390
     error rate: 0.7409330918379252 at iteration 3400
     error rate: 0.7409207186669672 at iteration 3410
     error rate: 0.740811160535605 at iteration 3420
     error rate: 0.7407362948277717 at iteration 3430
     error rate: 0.7406217780817109 at iteration 3440
     error rate: 0.7405498540379707 at iteration 3450
     error rate: 0.7404406634041977 at iteration 3460
     error rate: 0.7403496616710282 at iteration 3470
     error rate: 0.7402053810810151 at iteration 3480
     error rate: 0.7401572235558783 at iteration 3490
     error rate: 0.7399815692292996 at iteration 3500
     error rate: 0.7399220535463136 at iteration 3510
     error rate: 0.7397807096037284 at iteration 3520
     error rate: 0.7396787976697917 at iteration 3530
     error rate: 0.7395616687173914 at iteration 3540
     error rate: 0.7394588149184549 at iteration 3550
     error rate: 0.7393440963434503 at iteration 3560
     error rate: 0.7392878335384777 at iteration 3570
     error rate: 0.7391367127432302 at iteration 3580
     error rate: 0.7390871578333631 at iteration 3590
     error rate: 0.7389626396667474 at iteration 3600
     error rate: 0.7388986325404314 at iteration 3610
     error rate: 0.7387651549898399 at iteration 3620
     error rate: 0.7386541407445744 at iteration 3630
     error rate: 0.73851084620909 at iteration 3640
     error rate: 0.7384510744609433 at iteration 3650
     error rate: 0.7383179122253741 at iteration 3660
     error rate: 0.738230746790787 at iteration 3670
     error rate: 0.7381175547211153 at iteration 3680
     error rate: 0.7379998023427065 at iteration 3690
     error rate: 0.7378926484124746 at iteration 3700
     error rate: 0.7377524058610612 at iteration 3710
     error rate: 0.737629581382073 at iteration 3720
     error rate: 0.7375152747813201 at iteration 3730
     error rate: 0.737387862690449 at iteration 3740
     error rate: 0.7372320007039165 at iteration 3750
     error rate: 0.7371398335869486 at iteration 3760
     error rate: 0.7369726725597946 at iteration 3770
     error rate: 0.7368973312848651 at iteration 3780
     error rate: 0.7367083352196238 at iteration 3790
     error rate: 0.7366492315104832 at iteration 3800
     error rate: 0.7364964949276067 at iteration 3810
     error rate: 0.7363863648141186 at iteration 3820
     error rate: 0.7362238193697384 at iteration 3830
     error rate: 0.7361441943866089 at iteration 3840
     error rate: 0.7360135454201916 at iteration 3850
     error rate: 0.7359519779115347 at iteration 3860
     error rate: 0.7358498680941814 at iteration 3870
     error rate: 0.7357057514623498 at iteration 3880
     error rate: 0.7356304058469014 at iteration 3890
     error rate: 0.7355010523878 at iteration 3900
     error rate: 0.7353681104082309 at iteration 3910
     error rate: 0.7352312110632502 at iteration 3920
     error rate: 0.7351272445999826 at iteration 3930
     error rate: 0.7349818030090863 at iteration 3940
     error rate: 0.734874470289751 at iteration 3950
     error rate: 0.7347855625350462 at iteration 3960
     error rate: 0.7346775894996307 at iteration 3970
     error rate: 0.7345380422571035 at iteration 3980
     error rate: 0.7344579810778218 at iteration 3990
     error rate: 0.7343347983211962 at iteration 4000
     error rate: 0.7342609313706522 at iteration 4010
     error rate: 0.7341609015162077 at iteration 4020
     error rate: 0.7340559389416864 at iteration 4030
     error rate: 0.7339792693227899 at iteration 4040
     error rate: 0.7338769726151804 at iteration 4050
     error rate: 0.7337226991854702 at iteration 4060
     error rate: 0.7336460165870872 at iteration 4070
     error rate: 0.733518421826674 at iteration 4080
     error rate: 0.7334069618567453 at iteration 4090
     error rate: 0.73339590340065 at iteration 4100
     error rate: 0.7333089897758079 at iteration 4110
     error rate: 0.73322451446005 at iteration 4120
     error rate: 0.7331376564735991 at iteration 4130
     error rate: 0.7329816890893657 at iteration 4140
     error rate: 0.7328696974924424 at iteration 4150
     error rate: 0.7327180381888393 at iteration 4160
     error rate: 0.7326515975567061 at iteration 4170
     error rate: 0.732520238150395 at iteration 4180
     error rate: 0.7324287814595292 at iteration 4190
     error rate: 0.732315908560766 at iteration 4200
     error rate: 0.7322038495298583 at iteration 4210
     error rate: 0.7320691112836474 at iteration 4220
     error rate: 0.7320188609152161 at iteration 4230
     error rate: 0.7319449401646466 at iteration 4240
     error rate: 0.7318631281685876 at iteration 4250
     error rate: 0.7316823770150263 at iteration 4260
     error rate: 0.7316012326326051 at iteration 4270
     error rate: 0.7314706792253438 at iteration 4280
     error rate: 0.7313525064167796 at iteration 4290
     error rate: 0.7312489340473505 at iteration 4300
     error rate: 0.731121457158652 at iteration 4310
     error rate: 0.7309896119493252 at iteration 4320
     error rate: 0.7308470914760684 at iteration 4330
     error rate: 0.730770219727396 at iteration 4340
     error rate: 0.7306637205087315 at iteration 4350
     error rate: 0.7305577758658587 at iteration 4360
     error rate: 0.7304444377433311 at iteration 4370
     error rate: 0.7303032381118081 at iteration 4380
     error rate: 0.7301682287824748 at iteration 4390
     error rate: 0.7300646745490892 at iteration 4400
     error rate: 0.7299467304982744 at iteration 4410
     error rate: 0.7299260589702508 at iteration 4420
     error rate: 0.7297714165465821 at iteration 4430
     error rate: 0.7297181235677532 at iteration 4440
     error rate: 0.7295354793453686 at iteration 4450
     error rate: 0.7295074190428035 at iteration 4460
     error rate: 0.7293313400863483 at iteration 4470
     error rate: 0.7292636697204523 at iteration 4480
     error rate: 0.729100278236492 at iteration 4490
     error rate: 0.7290346643009554 at iteration 4500
     error rate: 0.7288943573013983 at iteration 4510
     error rate: 0.7287957501194767 at iteration 4520
     error rate: 0.7286940556937027 at iteration 4530
     error rate: 0.7285638944723123 at iteration 4540
     error rate: 0.728456273811295 at iteration 4550
     error rate: 0.728324629546113 at iteration 4560
     error rate: 0.7282838514424937 at iteration 4570
     error rate: 0.7281687076175885 at iteration 4580
     error rate: 0.728173990106399 at iteration 4590
     error rate: 0.7280019478510272 at iteration 4600
     error rate: 0.727945303060328 at iteration 4610
     error rate: 0.7278336577337776 at iteration 4620
     error rate: 0.7278363680250591 at iteration 4630
     error rate: 0.7277202976814154 at iteration 4640
     error rate: 0.727676628919234 at iteration 4650
     error rate: 0.7275580415812407 at iteration 4660
     error rate: 0.7275193951277992 at iteration 4670
     error rate: 0.7272942178123808 at iteration 4680
     error rate: 0.7273740082477339 at iteration 4690
     error rate: 0.7272004808360224 at iteration 4700
     error rate: 0.7272216386556868 at iteration 4710
     error rate: 0.7269944817299204 at iteration 4720
     error rate: 0.7270428131097773 at iteration 4730
     error rate: 0.7267739330565308 at iteration 4740
     error rate: 0.7267780943214571 at iteration 4750
     error rate: 0.7265606000179945 at iteration 4760
     error rate: 0.7266163882298323 at iteration 4770
     error rate: 0.7264867786912506 at iteration 4780
     error rate: 0.7264275900937849 at iteration 4790
     error rate: 0.7263409316177197 at iteration 4800
     error rate: 0.726297353767713 at iteration 4810
     error rate: 0.7260757782374765 at iteration 4820
     error rate: 0.7261568355591168 at iteration 4830
     error rate: 0.7259665936493135 at iteration 4840
     error rate: 0.7260874332887087 at iteration 4850
     error rate: 0.7259248099878767 at iteration 4860
     error rate: 0.7259925602330674 at iteration 4870
     error rate: 0.7257824649405548 at iteration 4880
     error rate: 0.7259769917776814 at iteration 4890
     error rate: 0.7257493306972305 at iteration 4900
     error rate: 0.7258464170214232 at iteration 4910
     error rate: 0.725639427279098 at iteration 4920
     error rate: 0.7258026619356115 at iteration 4930
     error rate: 0.7254305663160476 at iteration 4940
     error rate: 0.7256066559671924 at iteration 4950
     error rate: 0.7253618752642099 at iteration 4960
     error rate: 0.7255722709748017 at iteration 4970
     error rate: 0.7253681297417278 at iteration 4980
     error rate: 0.7254877602446004 at iteration 4990
    Total duration: 358.61 seconds
    


```python
# Takes the top 10% of values and maximizes. Rest are 0. 
percentile_animated_gif(sparsevc, output_filename='ExampleRPCA.gif',percentile_cutoff=96, frame_duration=5, image_size=(120, 160))
from IPython.display import Image, display
display(Image(filename='ExampleRPCA.gif'))
```

    GIF saved successfully as ExampleRPCA.gif.
    


    <IPython.core.display.Image object>


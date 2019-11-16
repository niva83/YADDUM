# Yet Another Dual-Doppler Uncertainty Model (YADDUM)
## Python package for dual-Doppler uncertainty assessment



[![DOI](https://zenodo.org/badge/221973907.svg)](https://zenodo.org/badge/latestdoi/221973907) <a href="https://www.buymeacoffee.com/z57lyJbHo" rel="nofollow"><img alt="https://img.shields.io/badge/Donate-Buy%20me%20a%20coffee-yellowgreen.svg" src="https://warehouse-camo.cmh1.psfhosted.org/1c939ba1227996b87bb03cf029c14821eab9ad91/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4275792532306d6525323061253230636f666665652d79656c6c6f77677265656e2e737667"></a>

Yes, true, all of us have their own uncertainty models! <br>
<br> However, when you need an operational uncertainty model (i.e., a piece of code that you can simply run) you cannot find one. Usually models out there are either buried deep inside some tools which are proprietary or people simply don't want to share their work. Alternatively, you are directed to read a bunch of hard to understand scientific papers and/or technical reports which have a large number of equations explained in somewhat dry, formal and not so understandable language. Therefore, the reproducibility of existing work isn't that great! <br>

But we don't want either of that, we want reproducible and reusable results which are freely shared and are explained in a simple way enabling not only understandable for academics but also to regular mortals. That's way I made this project. 

<br><br> This project is focused on delivering a simple yet effective dual-Doppler uncertainty model. It is based on the work of Michael Courtney and myself which has been presented on several occasions, e.g. :
<br> https://zenodo.org/record/1441178

Until the package becomes available through DTU conda chanel you can temporarily get it via Anaconda:
```
conda install -c nikola_v yaddum
```
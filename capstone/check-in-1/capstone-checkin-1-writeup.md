# **Capstone Check-in 1**

My capstone will involve Digital Signal Processing (DSP) in some way. I'll be taking audio data (.wav/.mp3 files) and generating data based on the signal. Many of the functions for extracting data from digital signals come from acoustics and physics. Examples include:

* [Fast Fourier Transform](https://www.nti-audio.com/en/support/know-how/fast-fourier-transform-fft) (FFT)
* [Mel-Frequency Cepstrum Coefficient](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) (MFCC)

---

## **Idea 1: Genre Classification**

### Goal

* Create a multiclass classification model that can predict the genre of a song based on a 30-second (or less) sample with better accuracy than the baseline model.

* Stretch Goal: Create playlists that are connected by extracted features of recorded sound instead of manually determined metadata for more fluid playlists that bounce between genre but fit the same sonic palette

### Data Sources

* Spotify scraping using `Spotipy` ([documentation link](https://spotipy.readthedocs.io/en/2.14.0/))
 
### Research Sources

* [Music Information Retrieval Website](https://musicinformationretrieval.com/) - resources on extracting information from music.

### Metrics

* Accuracy will be the most important metric here, since any misclassification is equally bad

### Audience

* Streaming services - build better playlists!
* DJs - create playlists with deeper cuts!
* Musicians - understand what an algorithm is looking for when writing your own music - get on more playlists!
* Adventurous Listeners - Experience algorithmic playlists that move between genres and introduce you to music you never knew you'd enjoy

---

## **Idea 2: Instrument Classification**

### Goal

* Create a multiclass classification model that can predict instrument type from a very short audio sample that only contains one instrument

* Stretch goal: Extract and classify individual instruments that are sounding simultaneously

* Stretch goal: Integrate this classifier with Reaper software to be able to automatically apply templates/track naming within the software depending on what instrument is playing on a track.

### Data Sources

* Sample Libraries (Native Instruments, MOTU, EastWest, Garritan)
* Scraping recordings of solo instruments from Spotify using `Spotipy`

### Research Sources

* [Music Information Retrieval Website](https://musicinformationretrieval.com/) - resources on extracting information from music.

* [Deep Learning for Audio Classification YT Playlist](https://www.youtube.com/watch?v=Z7YM-HAz-IY&list=PLhA3b2k8R3t2Ng1WW_7MiXeh1pfQJQi_P) - code-along resource for using deep learning to classify audio

### Metrics

* Accuracy will be the most important metric here, since any misclassification is equally bad

### Audience

* Audio Plug-in Developers - create machine-learning plug-ins that will speed up workflow
* DAW Developers - automate track creation templates based on instrument that is being recorded
---
## **Idea 3: Location Classification Based on Audio**

### Goal

* Create a multiclass classification model that can predict location (e.g. restaurant, park, office) based on a sample of the environmental noise

* Stretch Goal: Create a regression model that can estimate the relative population density of a location based on the environmental noise sample recorded at different points during a day.

### Data Sources

* Audio samples from soundsnap, storyblocks (large stock audio sites)

### Research Sources

* [Article from Landscape and Urban Planning journal](https://www.sciencedirect.com/science/article/pii/S0169204613000571) - may be behind paywall, I have access because I teach at a Univ.

### Metrics

* Accuracy will be the most important metric here, since any misclassification is equally bad

### Audience

* App developers - detecting location type based on sound could be helpful for users who are vision-impaired. might be less computationally intensive than computer vision.
* Aggregators - tell how busy a location is with more accuracy. Doesn't require individuals' location data
* Businesses - track the loudest/quietest moments in the day, could be used as a measure of traffic
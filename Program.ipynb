{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "* recording\n",
      "* done recordingg\n"
     ]
    }
   ],
   "source": [
    "#this program does the pitch shift to the record\n",
    "\n",
    "import pyaudio\n",
    "import wave\n",
    "import numpy as np\n",
    "\n",
    "#http://people.csail.mit.edu/hubert/pyaudio/\n",
    "#special thanks to http://zulko.github.io/blog/2014/03/29/soundstretching-and-pitch-shifting-in-python/\n",
    "\n",
    "FORMAT = pyaudio.paInt16\n",
    "CHANNELS = 2\n",
    "RATE = 44100\n",
    "\n",
    "#work with one huge chunk\n",
    "CHUNK = 204800\n",
    "RECORD_SECONDS = 5\n",
    "WAVE_OUTPUT_FILENAME = \"file.wav\"\n",
    "\n",
    "audio = pyaudio.PyAudio()\n",
    " \n",
    "# start Recording\n",
    "stream = audio.open(format=FORMAT, channels=CHANNELS,\n",
    "                rate=RATE, input=True,\n",
    "                frames_per_buffer=CHUNK)\n",
    "print (\"* recording\")\n",
    "\n",
    "def stretch(snd_array, factor, window_size, h):\n",
    "    \"\"\" Stretches/shortens a sound, by some factor. \"\"\"\n",
    "    phase = np.zeros(window_size)\n",
    "    hanning_window = np.hanning(window_size)\n",
    "    result = np.zeros(int(len(snd_array) / factor + window_size))\n",
    "    for i in np.arange(0, len(snd_array) - (window_size + h), h*factor):\n",
    "        i = int(i)\n",
    "        # Two potentially overlapping subarrays\n",
    "        a1 = snd_array[i: i + window_size]\n",
    "        a2 = snd_array[i + h: i + window_size + h]\n",
    "\n",
    "        # The spectra of these arrays\n",
    "        s1 = np.fft.fft(hanning_window * a1)\n",
    "        s2 = np.fft.fft(hanning_window * a2)\n",
    "\n",
    "        # Rephase all frequencies\n",
    "        phase = (phase + np.angle(s2/s1)) % 2*np.pi\n",
    "\n",
    "        a2_rephased = np.fft.ifft(np.abs(s2)*np.exp(1j*phase))\n",
    "        i2 = int(i/factor)\n",
    "        result[i2: i2 + window_size] += hanning_window*a2_rephased.real\n",
    "    return result.astype('int16')\n",
    "\n",
    "def speedx(sound_array, factor):\n",
    "    \"\"\" Multiplies the sound's speed by some `factor` \"\"\"\n",
    "    indices = np.round( np.arange(0, len(sound_array), factor) )\n",
    "    indices = indices[indices < len(sound_array)].astype(int)\n",
    "    return sound_array[ indices.astype(int) ]\n",
    "\n",
    "def pitchshift(snd_array, n, window_size=2**13, h=2**11):\n",
    "    \"\"\" Changes the pitch of a sound by ``n`` semitones. \"\"\"\n",
    "    factor = 2**(1.0 * n / 12.0)\n",
    "    stretched = stretch(snd_array, 1.0/factor, window_size, h)\n",
    "    return speedx(stretched[window_size:], factor)\n",
    "\n",
    "def playAudio(audio, samplingRate, channels):\n",
    "    p = pyaudio.PyAudio()\n",
    "    stream = p.open(format=pyaudio.paInt16,\n",
    "                    channels=channels,\n",
    "                    rate=samplingRate,\n",
    "                    output=True)\n",
    "    sound = (audio.astype(np.int16).tostring())\n",
    "    stream.write(sound)\n",
    "\n",
    "    stream.stop_stream()\n",
    "    stream.close()\n",
    "    p.terminate()\n",
    "    return\n",
    "\n",
    "data = stream.read(CHUNK)\n",
    "data = np.fromstring(data, dtype=np.int16)\n",
    "\n",
    "#make two times louder\n",
    "data *= 2\n",
    "\n",
    "print (\"* done recording\")\n",
    "\n",
    "# stop Recording\n",
    "stream.stop_stream()\n",
    "stream.close()\n",
    "audio.terminate()\n",
    "\n",
    "#Tests\n",
    "playAudio(data, RATE, CHANNELS)\n",
    "\n",
    "pitched = pitchshift(data, -5)\n",
    "playAudio(pitched, RATE, CHANNELS)\n",
    "\n",
    "pitched = pitchshift(data, 5)\n",
    "playAudio(pitched, RATE, CHANNELS)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WAV file was saved as file.wav\n"
     ]
    }
   ],
   "source": [
    "#save file\n",
    "waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')\n",
    "waveFile.setnchannels(CHANNELS)\n",
    "waveFile.setsampwidth(audio.get_sample_size(FORMAT))\n",
    "waveFile.setframerate(RATE)\n",
    "waveFile.writeframes(b''.join(data))\n",
    "waveFile.close()\n",
    "print('WAV file was saved as', WAVE_OUTPUT_FILENAME)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

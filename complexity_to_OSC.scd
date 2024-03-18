//Receives coefficients from a python script via Open Sound Control and uses values as variables within digital sound synthesizers
s.options.sampleRate = 41000
s.boot;
s.option
s.options.numInputBusChannels_(0);
ServerOptions.outDevices;
"open -a 'Audio MIDI Setup'".unixCmd;

~mixer = NdefMixer(s);
ProxyMeter.addMixer(~mixer);
s.plotTree
Ndef.clear(10)
Ndef.clear(1);

// Create simple test Ndef
(
Ndef(\testSynth, {
	arg pulse = 200;
	/*pulse = VarLag.kr(pulse, 2);*/
	Pulse.ar(pulse.poll(1));

}).add;
)


(
Ndef(\testSine, {
	arg freq = 200;
	/*pulse = VarLag.kr(pulse, 2);*/
	SinOsc.ar(freq.poll(1));

}).add;
)


// Test manual value change
Ndef(\testSine).fadeTime = 2;
Ndef(\testSine).play;
Ndef(\testSine).xset(\freq, 500);

// Set up OSC to change paramenter inside Ndef
(
OSCdef(key: \myOscDef, func: {|msg|
    //# msg[0] is the path that the message was received on
	Ndef(\testSine).xset(\freq, msg[1]/2);
}, path: "/my_path");
)


// Test manual value change
Ndef(\testSynth).fadeTime = 2;
Ndef(\testSynth).play;
Ndef(\testSynth).xset(\pulse, 300);

Ndef(\testSynth).release;

// Set up OSC to change paramenter inside Ndef
(
OSCdef(key: \myOscDef, func: {|msg|
    //# msg[0] is the path that the message was received on
	Ndef(\testSynth).xset(\pulse, msg[1]);
}, path: "/my_path");
)


//// Load and play audio files
~path = '/Users/andreas/Desktop/thesis playgrund/Rundgang/say_a_little_prayer.wav'
(
~buf1 = Buffer.read(s,~path);
Ndef(\prayer, {arg mul = 0.5;
	PlayBuf.ar(2, ~buf1, loop: 50) * mul.poll(1)
});
)
Ndef(\prayer).play;


// Set up OSC to change paramenter inside Ndef
(
OSCdef(key: \myOscDef, func: {|msg|
    //# msg[0] is the path that the message was received on
	Ndef(\prayer).xset(\mul, msg[1]/100);
}, path: "/my_path");
)
///////////////////////////////////////////////////////////////////
// reverberated sine percussion
(
Ndef(\dropsRev, {arg vol = 1;
var a, c, d, s, z, y;
d = 2; // number of percolators
c = 2; // number of comb delays
a = 1; // number of allpass delays

({
		// sine percolation sound :
	s = Mix.ar(Array.fill(d, { Resonz.ar(Dust.ar(2/d, 50), 200 + 3000.0.rand, 0.003)}) );

		// reverb predelay time :
	z = DelayN.ar(s, 0.048);

		// 7 length modulated comb delays in parallel :
	y = Mix.ar(CombL.ar(z, 0.1, LFNoise1.kr(Array.fill(c,{0.1.rand}), 0.04, 0.05), 15));

		// chain of 4 allpass delays on each of two channels (8 total) :
	a.do({ y = AllpassN.ar(y, 0.050, [0.050.rand,0.050.rand], 1) });
		// add original sound to reverb and play it :
		(s+(0.2*y))*vol.poll(1)
})}).add;
)

// Test manual value change
Ndef(\dropsRev).fadeTime = 2;
Ndef(\dropsRev).play;
Ndef(\dropsRev).set(\vol, 0.5);

Ndef(\dropsRev).release;


// Set up OSC to change paramenter inside Ndef
(
OSCdef(key: \myOscDef, func: {|msg|
    //# msg[0] is the path that the message was received on
	// Ndef(\dropsRev).xset(\vol, msg[3]);
	Ndef(\dropsRev).xset(\vol, msg[1]);
	// Ndef(\glassDroplets).xset(\centerFreq, msg[2]);
}, path: "/my_path");
)



///////////////////////////////////////////////////////////////////
(
Ndef(\glassDroplets,
	// sample and hold liquidities
	// mouse x controls clock rate, mouse y controls center frequency
{
	arg clockRate = 3, clockTime, clock, centerFreq = 300, freq, panPos, patch;

		/*clockRate = MouseX.kr(1, 200, 'exponential').poll(1);*/
		clockTime = clockRate.reciprocal;
	clock = Impulse.kr(clockRate.poll(1), 0.4);

		/*centerFreq = MouseY.kr(100, 8000, 'exponential');*/
	freq = Latch.kr(WhiteNoise.kr(centerFreq.poll(1) * 0.5, centerFreq), clock);
	panPos = Latch.kr(WhiteNoise.kr, clock);
	patch = CombN.ar(
			Pan2.ar(
				SinOsc.ar(
					freq,
					0,
					Decay2.kr(clock, 0.1 * clockTime, 0.9 * clockTime)
				),
				panPos
			),
			0.3, 0.3, 2
		);
}).add;
)

Ndef(\glassDroplets).fadeTime = 2;
Ndef(\glassDroplets).play;
Ndef(\glassDroplets).release(1);
// Test manual value change
Ndef(\glassDroplets).set(\clockRate, 60);
Ndef(\glassDroplets).set(\centerFreq, 100);

// Set up OSC to change paramenter inside Ndef
(
OSCdef(key: \myOscDef, func: {|msg|
    //# msg[0] is the path that the message was received on
	Ndef(\glassDroplets).set(\clockRate, msg[1]);
	Ndef(\glassDroplets).set(\centerFreq, msg[2]);
}, path: "/my_path");
)
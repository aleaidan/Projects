// =======================================================
// Interactive Desktop Audio System
// - Background loop with low-pass filter (Y-axis)
// - Reverb + stereo width (X-axis)
// - Left click plays .wav sound effect (dry / unfiltered)
// =======================================================
(
Server.default.options.numInputBusChannels = 0;
Server.default.options.sampleRate = 44100;

s.waitForBoot {
    ~loopFile = "/Users/alexsuarez/desktop/MAP/music assets/background_loop.wav";
    ~clickFile = "/Users/alexsuarez/desktop/MAP/music assets/click_sfx.wav";

    ~loopBuf = Buffer.read(s, ~loopFile);
    ~clickBuf = Buffer.read(s, ~clickFile);

    SynthDef(\loopPlayer, {
        |bufnum = 0, lpFreq = 1000, revMix = 0.2, width = 0.5|
        var sig, dry, wet;

        sig = PlayBuf.ar(2, bufnum, BufRateScale.kr(bufnum), loop: 1);
        sig = RLPF.ar(sig, Lag.kr(lpFreq, 0.1), 0.3);

        dry = sig;

        // Substitute JPverb with FreeVerb
        wet = FreeVerb.ar(sig, mix: revMix, room: 0.8, damp: 0.3);

        // Apply stereo width
        wet = Splay.ar(wet, Lag.kr(width, 0.2), center: 0);

        sig = XFade2.ar(dry, wet, (revMix * 2) - 1);

        Out.ar(0, sig * 0.5);
    }).add;

    SynthDef(\clickSFX, {
        |bufnum = 0|
        var sig = PlayBuf.ar(2, bufnum, BufRateScale.kr(bufnum), doneAction: 2);
        Out.ar(0, sig * 0.6);
    }).add;

    s.sync;

    ~loop = Synth(\loopPlayer, [\bufnum, ~loopBuf.bufnum]);
    ~lastX = 0.5; ~lastY = 0.5;

    OSCdef(\cursorTracker, { |msg, time, addr|
    var x, y, lpFreq, revMix, width;

    x = msg[1].clip(0, 1);
    y = msg[2].clip(0, 1);

    ~lastX = x;
    ~lastY = y;

    // Y → filter cutoff (100 Hz - 10 kHz)
    lpFreq = y.linexp(0, 1, 10000, 100);

    // X → reverb depth (subtle) + stereo width
    revMix = x.linexp(0, 1, 0.01, 0.6);  // more dynamic taper
    width  = x.linlin(0, 1, 0.0, 1.0);

    if (~loop.notNil) {
        ~loop.set(\lpFreq, lpFreq, \revMix, revMix, \width, width);
    };

    ("Cursor OSC — x:" + x.round(0.01)
     + " y:" + y.round(0.01)
     + " → lpFreq:" + lpFreq.round(1)
     + " revMix:" + revMix.round(0.01)
     + " width:" + width.round(0.01)).postln;

}, '/cursor');


    OSCdef(\clickSfx, { |msg, time, addr|
        "Click SFX triggered.".postln;
        Synth(\clickSFX, [\bufnum, ~clickBuf.bufnum]);
    }, '/play_sfx');

    "SuperCollider OSC system ready with reverb + stereo width.".postln;
};
)

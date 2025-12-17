import streamlit as st
import base64

def get_player_js():
    """
    Returns the JavaScript code to initialize the global audio player.
    Includes a visual Interface for progress and buffering.
    """
    return """
    <style>
        #audio-player-container {
            font-family: sans-serif;
            background: #f1f3f6;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }
        #play-pause-btn {
            background: #ff4b4b;
            color: white;
            border: none;
            border-radius: 50%;
            width: 32px;
            height: 32px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
        }
        #timeline-container {
            flex-grow: 1;
            height: 6px;
            background: #d0d0d0;
            border-radius: 3px;
            position: relative;
            cursor: pointer;
            overflow: hidden;
        }
        #buffered-bar {
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            background: #a0a0a0;
            width: 0%;
            transition: width 0.2s;
        }
        #played-bar {
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            background: #ff4b4b;
            width: 0%;
            transition: width 0.1s linear;
        }
        #time-display {
            font-size: 12px;
            color: #555;
            min-width: 80px;
            text-align: right;
            font-variant-numeric: tabular-nums;
        }
    </style>

    <div id="audio-player-container">
        <button id="play-pause-btn" onclick="window.top.myGlobalAudioPlayer.togglePlay()">▶</button>
        <div id="timeline-container">
            <div id="buffered-bar"></div>
            <div id="played-bar"></div>
        </div>
        <div id="time-display">00:00</div>
    </div>

    <script>
    (function() {
        // Just ensures the elements are available in the top window if we want to persist UI?
        // Actually, Streamlit re-renders HTML. We need to attach the Logic to window.top, 
        // but the UI is fresh. We must sync UI to global state.
        
        const container = document.getElementById('audio-player-container');
        const playBtn = document.getElementById('play-pause-btn');
        const playedBar = document.getElementById('played-bar');
        const bufferedBar = document.getElementById('buffered-bar');
        const timeDisplay = document.getElementById('time-display');

        if (!window.top.myGlobalAudioPlayer) {
            console.log("[AudioPlayer] Initializing global player...");
            
            window.top.myGlobalAudioPlayer = {
                queue: [], // Array of {url, duration}
                history: [], // Array of {url, duration} - played chunks
                currentChunk: null, // {url, duration}
                audioObj: new Audio(),
                isPlaying: false,
                
                // Stats
                cumulativePlayed: 0, 
                totalBuffered: 0,
                
                enqueue: function(b64data) {
                    const blob = this.b64toBlob(b64data, "audio/wav");
                    const url = URL.createObjectURL(blob);
                    
                    // Create temp audio to get duration
                    const tempAudio = new Audio(url);
                    tempAudio.onloadedmetadata = () => {
                        const duration = tempAudio.duration;
                        this.totalBuffered += duration;
                        this.queue.push({url: url, duration: duration});
                        this.updateUI();
                        
                        if (!this.currentChunk && !this.isPlaying) {
                            this.playNext();
                        }
                    };
                },
                
                playNext: function() {
                    if (this.queue.length === 0) {
                        this.isPlaying = false;
                        this.currentChunk = null;
                        this.updateUI();
                        return;
                    }
                    
                    this.isPlaying = true;
                    this.currentChunk = this.queue.shift();
                    
                    this.audioObj.src = this.currentChunk.url;
                    this.audioObj.play().catch(e => console.error("Playback error:", e));
                    
                    this.audioObj.onended = () => {
                        this.cumulativePlayed += this.currentChunk.duration;
                        this.history.push(this.currentChunk);
                        // Clean up object URL? Maybe keep for seek (not implemented yet)
                        // URL.revokeObjectURL(this.currentChunk.url); 
                        this.playNext();
                    };
                    
                    this.updateUI();
                },
                
                togglePlay: function() {
                    if (this.isPlaying) {
                        this.audioObj.pause();
                        this.isPlaying = false;
                    } else {
                        if (this.currentChunk) {
                             this.audioObj.play();
                             this.isPlaying = true;
                        } else {
                            this.playNext();
                        }
                    }
                    this.updateUI();
                },
                
                b64toBlob: function(b64Data, contentType) {
                    const byteCharacters = atob(b64Data);
                    const byteArrays = [];
                    for (let offset = 0; offset < byteCharacters.length; offset += 512) {
                        const slice = byteCharacters.slice(offset, offset + 512);
                        const byteNumbers = new Array(slice.length);
                        for (let i = 0; i < slice.length; i++) {
                            byteNumbers[i] = slice.charCodeAt(i);
                        }
                        const byteArray = new Uint8Array(byteNumbers);
                        byteArrays.push(byteArray);
                    }
                    return new Blob(byteArrays, {type: contentType});
                },

                updateLoop: function() {
                    // Update Time Display and Bars
                    if (!this.currentChunk) return;
                    
                    const currentChunkTime = this.audioObj.currentTime;
                    const totalCurrentTime = this.cumulativePlayed + currentChunkTime;
                    
                    // Update UI Elements if they exist (need to re-bind since streamlit reruns)
                     // Note: The UI elements in `window.top` context vs iframe context.
                     // The script runs in iframe. The UI is in iframe.
                     // But the `myGlobalAudioPlayer` is in `window.top`.
                     // We need a way to call "back" into the current iframe's UI.
                },
                
                reset: function() {
                    this.audioObj.pause();
                    this.queue = [];
                    this.history = [];
                    this.currentChunk = null;
                    this.isPlaying = false;
                    this.cumulativePlayed = 0;
                    this.totalBuffered = 0;
                }
            };
        }
        
        // --- Connection Logic (Iframe <-> Global) ---
        const player = window.top.myGlobalAudioPlayer;
        
        // Update function specific to THIS render
        function updateVisuals() {
            if (!player) return;
            
            // Sync Play Button
            playBtn.innerText = player.isPlaying ? "⏸" : "▶";
            
            // Calculate Times
            let currentT = player.cumulativePlayed;
            if (player.currentChunk) {
                currentT += player.audioObj.currentTime;
            }
            
            const totalT = player.totalBuffered > 0 ? player.totalBuffered : 1; // avoid /0
            
            // Format Time
            const m = Math.floor(currentT / 60).toString().padStart(2, '0');
            const s = Math.floor(currentT % 60).toString().padStart(2, '0');
            timeDisplay.textContent = `${m}:${s}`;
            
            // Update Bars
            // Since we want "growing" bar:
            // "Buffered" should effectively fill 100% of the visible container if we treat container as "Total Buffered So Far"
            // But usually container = Total Expected duration. We don't know it.
            // Let's make the container represent "Total Buffered".
            
            const pPct = (currentT / totalT) * 100;
            playedBar.style.width = `${pPct}%`;
            bufferedBar.style.width = `100%`; // Until we assume a larger total, buffered is everything we have.
            
            requestAnimationFrame(updateVisuals);
        }
        
        // Hook into global updates (optional)
        player.updateUI = function() {
             // trigger explicit updates if needed
        };
        
        // Start Loop
        updateVisuals();
        
    })();
    </script>
    """

def init_player():
    """Injects the player initialization script."""
    st.components.v1.html(get_player_js(), height=60, width=None)

def enqueue_audio(file_path):
    """
    Reads a wav file, converts to base64, and injects a script to push it to the player queue.
    """
    try:
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode('utf-8')
            
        js = f"""
        <script>
        if (window.top.myGlobalAudioPlayer) {{
            window.top.myGlobalAudioPlayer.enqueue("{b64}");
        }}
        </script>
        """
        # We use a unique key or just allow it to run
        st.components.v1.html(js, height=0, width=0)
    except Exception as e:
        print(f"Error enqueueing audio: {e}")

def reset_player():
    """Resets the global player state."""
    js = """
    <script>
    if (window.top.myGlobalAudioPlayer) {
        window.top.myGlobalAudioPlayer.reset();
    }
    </script>
    """
    st.components.v1.html(js, height=0, width=0)

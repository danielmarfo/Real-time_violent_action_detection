const video = document.getElementById("video");
const placeholder = document.getElementById("video-placeholder");
const alarm = document.getElementById("alarm-sound");
const monitorBtn = document.getElementById("monitor-btn");
const stopBtn = document.getElementById("stop-alarm");
const threatLevelEl = document.getElementById("threat-level");
const monitoringStatus = document.getElementById("monitoring-status");

let monitoring = false;
let statusInterval = null;

// Toggle monitoring (Start ↔ Stop)
monitorBtn.addEventListener("click", () => {
  if (!monitoring) {
    // Start monitoring
    monitoring = true;
    monitoringStatus.textContent = "Active";
    monitoringStatus.classList.remove("inactive");
    monitoringStatus.classList.add("active");

    video.src = "/video_feed";
    video.style.display = "block";
    placeholder.style.display = "none";

    monitorBtn.textContent = "⏹ Stop Monitoring";
    monitorBtn.classList.remove("green");
    monitorBtn.classList.add("red");

    statusInterval = setInterval(checkStatus, 1000);
  } else {
    // Stop monitoring
    monitoring = false;
    monitoringStatus.textContent = "Inactive";
    monitoringStatus.classList.remove("active");
    monitoringStatus.classList.add("inactive");

    video.src = "";
    video.style.display = "none";
    placeholder.style.display = "block";

    monitorBtn.textContent = "▶ Start Monitoring";
    monitorBtn.classList.remove("red");
    monitorBtn.classList.add("green");

    clearInterval(statusInterval);
    statusInterval = null;

    // Also stop alarm if playing
    alarm.pause();
    alarm.currentTime = 0;
    stopBtn.disabled = true;
  }
});


let muteUntil = 0; // ⬅️ track mute time

// Stop alarm manually
stopBtn.addEventListener("click", () => {
  alarm.pause();
  alarm.currentTime = 0;
  stopBtn.disabled = true;

  // ⬅️ set mute for 10 seconds
  muteUntil = Date.now() + 10000;
});

// Poll backend status
async function checkStatus() {
  try {
    const res = await fetch("/status");
    const data = await res.json();

    const threatPercent = (data.threat * 100).toFixed(1) + "%";
    threatLevelEl.textContent = threatPercent;

    if (data.violence) {
      // ⬅️ Only play alarm if not muted
      if (Date.now() > muteUntil) {
        alarm.loop = true;
        alarm.play();
        stopBtn.disabled = false;
      }
    }
  } catch (err) {
    console.error("Status check failed:", err);
  }
}


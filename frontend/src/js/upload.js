document.getElementById("uploadBtn").addEventListener("click", async () => {
  console.log("🚀 Upload button clicked!");

  const answerFile = document.getElementById("answer").files[0];
  const worksheetFile = document.getElementById("worksheet").files[0];
  const assignmentType = document.getElementById("assignmentType").value;

  if (!answerFile || !worksheetFile) {
    alert("❌ Please select both an answer key and a worksheet PDF.");
    return;
  }

  console.log(`📂 Selected files: ${answerFile.name}, ${worksheetFile.name}`);

  const formData = new FormData();
  formData.append("assignmentType", assignmentType);
  formData.append("answer", answerFile);
  formData.append("worksheet", worksheetFile);

  // Show the loading overlay
  document.getElementById("loadingOverlay").style.display = "flex";

  try {
    const response = await fetch("http://127.0.0.1:8000/grade-pdfs", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`❌ Server error: ${response.status}`);
    }

    console.log("✅ Upload successful!");
  } catch (error) {
    console.error("❌ Upload failed:", error);
    alert("Upload failed. Please try again."); // Show error message to user
  } finally {
    // Hide the loading overlay and redirect to dashboard
    document.getElementById("loadingOverlay").style.display = "none";
    window.location.href = "pages/dashboard.html";
  }
});

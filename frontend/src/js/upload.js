document.getElementById("uploadBtn").addEventListener("click", async () => {
    console.log("🚀 Upload button clicked!");

    const answerFile = document.getElementById("answer").files[0];
    const worksheetFile = document.getElementById("worksheet").files[0];
    const pdfContainer = document.getElementById("pdfContainer");
    const pdfViewer = document.getElementById("pdfViewer");

    if (!answerFile || !worksheetFile) {
        alert("❌ Please select both an answer key and a worksheet PDF.");
        return;
    }

    // Hide the PDF container before uploading
    pdfContainer.style.display = "none";

    console.log(`📂 Selected files: ${answerFile.name}, ${worksheetFile.name}`);

    const formData = new FormData();
    formData.append("answer", answerFile);
    formData.append("worksheet", worksheetFile);

    console.log("📤 Sending files to FastAPI...");

    try {
        const response = await fetch("http://127.0.0.1:8000/grade-pdfs", {
            method: "POST",
            body: formData
        });

        console.log("📥 Response received:", response);

        if (!response.ok) {
            throw new Error(`❌ Server error: ${response.status}`);
        }

        console.log("✅ Upload successful!");

        // Now fetch and display the graded PDF
        fetchGradedPDF();

    } catch (error) {
        console.error("❌ Upload failed:", error);
    }
});

async function fetchGradedPDF() {
    console.log("📥 Fetching graded PDF...");

    try {
        const response = await fetch("http://127.0.0.1:8000/get-graded-pdf");

        if (!response.ok) {
            throw new Error("❌ No graded PDF found.");
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);

        console.log("✅ Graded PDF received!");

        // ✅ Dynamically update the webpage
        const pdfContainer = document.getElementById("pdfContainer");
        const pdfViewer = document.getElementById("pdfViewer");

        pdfViewer.src = url;  // Set the `iframe` source
        pdfContainer.style.display = "block"; // Show the container

    } catch (error) {
        console.error("❌ Failed to fetch graded PDF:", error);
    }
}

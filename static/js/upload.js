document.getElementById("file_peptides").addEventListener('change', function selectedFileChanged() {
    if (this.files.length === 0) {
        return;
    }

    const file = this.files[0];

    const reader = new FileReader();
    reader.onload = function fileReadCompleted() {
        // when the reader is done, the content is in reader.result.
        document.getElementById("peptides").value = reader.result;
    };
    reader.readAsText(file);
});

document.getElementById("submission_form").addEventListener("submit", function () {
    document.body.style.cursor = "wait";
    const allElements = document.querySelectorAll('*');
    allElements.forEach(element => {
        element.style.cursor = 'wait';
    });
    document.getElementById("btn_submit").disabled = true;
})
// Column Search Functionality
// This script adds search/filter functionality to the column selection lists

document.addEventListener('DOMContentLoaded', () => {
    const xSearch = document.getElementById('xSearch');
    const ySearch = document.getElementById('ySearch');
    const xSelect = document.getElementById('xSelect');
    const ySelect = document.getElementById('ySelect');

    // Filter function
    function filterColumnList(searchInput, columnList) {
        const searchTerm = searchInput.value.toLowerCase().trim();
        const columnItems = Array.from(columnList.children);

        columnItems.forEach(item => {
            // Only filter column-item divs, not other elements
            if (item.classList.contains('column-item')) {
                const columnName = item.textContent.toLowerCase();
                if (columnName.includes(searchTerm)) {
                    item.style.display = '';
                } else {
                    item.style.display = 'none';
                }
            }
        });
    }

    // Add event listeners
    if (xSearch && xSelect) {
        xSearch.addEventListener('input', () => filterColumnList(xSearch, xSelect));
    }

    if (ySearch && ySelect) {
        ySearch.addEventListener('input', () => filterColumnList(ySearch, ySelect));
    }
});

/**
 * TableOfContents - A class for managing dynamic table of contents generation and navigation
 * Generates a table of contents from heading elements and provides smooth scrolling navigation
 */
class TableOfContents {
  /**
   * Initialize the TableOfContents instance
   * @param {Object} options - Configuration options
   * @param {string} options.contentSelector - CSS selector for the content container
   * @param {string} options.tocSelector - CSS selector for the TOC container
   * @param {string} options.headingSelector - CSS selector for heading elements (default: 'h1, h2, h3, h4, h5, h6')
   * @param {number} options.scrollOffset - Offset in pixels for smooth scrolling (default: 0)
   * @param {boolean} options.smoothScroll - Enable smooth scrolling (default: true)
   * @param {Array<string>} options.excludeHeadings - Classes to exclude from TOC (default: [])
   */
  constructor(options = {}) {
    this.contentSelector = options.contentSelector || 'main';
    this.tocSelector = options.tocSelector || '.toc';
    this.headingSelector = options.headingSelector || 'h1, h2, h3, h4, h5, h6';
    this.scrollOffset = options.scrollOffset || 0;
    this.smoothScroll = options.smoothScroll !== false;
    this.excludeHeadings = options.excludeHeadings || [];

    this.contentElement = null;
    this.tocElement = null;
    this.headings = [];
    this.activeLink = null;

    this.init();
  }

  /**
   * Initialize the table of contents
   */
  init() {
    this.contentElement = document.querySelector(this.contentSelector);
    this.tocElement = document.querySelector(this.tocSelector);

    if (!this.contentElement || !this.tocElement) {
      console.error('TableOfContents: Content or TOC element not found');
      return;
    }

    this.extractHeadings();
    this.generateTOC();
    this.attachEventListeners();
  }

  /**
   * Extract all headings from the content element
   */
  extractHeadings() {
    const allHeadings = this.contentElement.querySelectorAll(this.headingSelector);

    this.headings = Array.from(allHeadings).filter(heading => {
      // Skip headings with excluded classes
      return !this.excludeHeadings.some(excludeClass =>
        heading.classList.contains(excludeClass)
      );
    });

    // Assign IDs to headings that don't have them
    this.headings.forEach((heading, index) => {
      if (!heading.id) {
        heading.id = `heading-${index}`;
      }
    });
  }

  /**
   * Generate the table of contents HTML
   */
  generateTOC() {
    if (this.headings.length === 0) {
      this.tocElement.innerHTML = '<p>No headings found</p>';
      return;
    }

    const tocList = this.createTOCList(this.headings);
    this.tocElement.innerHTML = '';
    this.tocElement.appendChild(tocList);
  }

  /**
   * Create a nested list structure from headings
   * @param {Array<HTMLElement>} headings - Array of heading elements
   * @returns {HTMLElement} - UL element containing the TOC structure
   */
  createTOCList(headings) {
    const ul = document.createElement('ul');
    let currentLevel = this.getHeadingLevel(headings[0]);
    let currentList = ul;
    const listStack = [{ level: currentLevel, list: ul }];

    headings.forEach(heading => {
      const level = this.getHeadingLevel(heading);
      const li = document.createElement('li');

      const link = document.createElement('a');
      link.href = `#${heading.id}`;
      link.textContent = heading.textContent;
      link.className = 'toc-link';

      li.appendChild(link);

      // Handle nested levels
      if (level > currentLevel) {
        // Create new nested lists for deeper levels
        for (let i = currentLevel; i < level; i++) {
          const newUl = document.createElement('ul');
          if (currentList.lastElementChild) {
            currentList.lastElementChild.appendChild(newUl);
          } else {
            currentList.appendChild(newUl);
          }
          listStack.push({ level: i + 1, list: newUl });
          currentList = newUl;
        }
      } else if (level < currentLevel) {
        // Go back up to the appropriate level
        while (listStack.length > 1 && listStack[listStack.length - 1].level > level) {
          listStack.pop();
        }
        currentList = listStack[listStack.length - 1].list;
      }

      currentList.appendChild(li);
      currentLevel = level;

      // Attach click handler for smooth scrolling
      link.addEventListener('click', (e) => this.handleLinkClick(e, heading));
    });

    return ul;
  }

  /**
   * Get the heading level (1-6)
   * @param {HTMLElement} heading - Heading element
   * @returns {number} - Heading level
   */
  getHeadingLevel(heading) {
    return parseInt(heading.tagName[1], 10);
  }

  /**
   * Handle TOC link click events
   * @param {Event} e - Click event
   * @param {HTMLElement} heading - Target heading element
   */
  handleLinkClick(e, heading) {
    e.preventDefault();

    // Update active link
    this.setActiveLink(e.target);

    // Scroll to heading
    this.scrollToElement(heading);
  }

  /**
   * Set the active TOC link
   * @param {HTMLElement} link - Link element to mark as active
   */
  setActiveLink(link) {
    if (this.activeLink) {
      this.activeLink.classList.remove('active');
    }
    link.classList.add('active');
    this.activeLink = link;
  }

  /**
   * Scroll to an element with optional offset
   * @param {HTMLElement} element - Element to scroll to
   */
  scrollToElement(element) {
    const targetPosition = element.offsetTop - this.scrollOffset;

    if (this.smoothScroll) {
      window.scrollTo({
        top: targetPosition,
        behavior: 'smooth'
      });
    } else {
      window.scrollTo(0, targetPosition);
    }
  }

  /**
   * Attach event listeners for scroll tracking
   */
  attachEventListeners() {
    // Track scroll position and update active link
    window.addEventListener('scroll', () => this.updateActiveLink());

    // Regenerate TOC on window resize if needed
    window.addEventListener('resize', () => {
      // Optional: re-extract headings in case content changed
    });

    // Listen for dynamic content changes
    const observer = new MutationObserver(() => {
      this.extractHeadings();
      this.generateTOC();
    });

    observer.observe(this.contentElement, {
      childList: true,
      subtree: true
    });
  }

  /**
   * Update the active link based on current scroll position
   */
  updateActiveLink() {
    const scrollPosition = window.scrollY + this.scrollOffset;
    let activeHeading = null;

    // Find the current heading in view
    for (let i = this.headings.length - 1; i >= 0; i--) {
      if (this.headings[i].offsetTop <= scrollPosition) {
        activeHeading = this.headings[i];
        break;
      }
    }

    if (activeHeading) {
      const link = this.tocElement.querySelector(`a[href="#${activeHeading.id}"]`);
      if (link && link !== this.activeLink) {
        this.setActiveLink(link);
      }
    }
  }

  /**
   * Destroy the TableOfContents instance and clean up
   */
  destroy() {
    window.removeEventListener('scroll', () => this.updateActiveLink());
    this.tocElement.innerHTML = '';
    this.headings = [];
    this.activeLink = null;
  }

  /**
   * Refresh the table of contents
   */
  refresh() {
    this.extractHeadings();
    this.generateTOC();
  }

  /**
   * Get the current headings
   * @returns {Array<HTMLElement>} - Array of heading elements
   */
  getHeadings() {
    return this.headings;
  }

  /**
   * Navigate to a specific heading by index
   * @param {number} index - Index of the heading to navigate to
   */
  navigateToHeading(index) {
    if (index >= 0 && index < this.headings.length) {
      this.scrollToElement(this.headings[index]);
    }
  }

  /**
   * Navigate to a specific heading by ID
   * @param {string} headingId - ID of the heading to navigate to
   */
  navigateToHeadingById(headingId) {
    const heading = document.getElementById(headingId);
    if (heading) {
      this.scrollToElement(heading);
    }
  }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = TableOfContents;
}

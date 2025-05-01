var structuredData = {
  "@context": "https://schema.org",
  "@type": "WebSite",
  "name": "FLP0478 - PLN4HUM",
  "alternateName": "PLN4HUM",
  "url": "https://github.com/PedroSchmalz/FLP0478",
  "potentialAction": {
    "@type": "SearchAction",
    "target": {
      "@type": "EntryPoint",
      "urlTemplate": "https://github.com/PedroSchmalz/FLP0478/search?q={search_term_string}"
    },
    "query-input": "required name=search_term_string"
  }
};

const SDscript = document.createElement('script');
SDscript.setAttribute('type', 'application/ld+json');
SDscript.textContent = JSON.stringify(structuredData);
document.head.appendChild(SDscript);
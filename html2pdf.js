const { chromium } = require('/home/kyle/tools/node_modules/playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('file:///home/kyle/schwa_spgc/paper.html', { waitUntil: 'networkidle' });
  await page.pdf({
    path: '/home/kyle/schwa_spgc/paper.pdf',
    format: 'Letter',
    margin: { top: '0.75in', bottom: '0.75in', left: '1in', right: '1in' },
    printBackground: true
  });
  await browser.close();
  console.log('PDF written');
})();

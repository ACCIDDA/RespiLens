import { Helmet } from "react-helmet-async";
import { useLocation } from "react-router-dom";

const SITE_URL = (
  import.meta.env.VITE_SITE_URL || "https://www.respilens.com"
).replace(/\/$/, "");

const DEFAULT_DESCRIPTION =
  "Explore respiratory disease forecasts, surveillance data, and interactive tools for influenza, COVID-19, and RSV across the United States.";
const DEFAULT_IMAGE = "/og-image.png";
const DEFAULT_IMAGE_ALT =
  "RespiLens preview showing respiratory disease forecasts and surveillance data.";

const toAbsoluteUrl = (pathOrUrl) => {
  if (!pathOrUrl) return `${SITE_URL}${DEFAULT_IMAGE}`;
  if (/^https?:\/\//i.test(pathOrUrl)) return pathOrUrl;
  return `${SITE_URL}${pathOrUrl.startsWith("/") ? pathOrUrl : `/${pathOrUrl}`}`;
};

const Seo = ({
  title = "RespiLens",
  description = DEFAULT_DESCRIPTION,
  canonicalPath,
  image = DEFAULT_IMAGE,
  imageAlt = DEFAULT_IMAGE_ALT,
  type = "website",
  structuredData,
}) => {
  const location = useLocation();
  const path = canonicalPath || location.pathname || "/";
  const canonicalUrl = toAbsoluteUrl(path);
  const imageUrl = toAbsoluteUrl(image);
  const pageTitle = title.includes("RespiLens")
    ? title
    : `${title} | RespiLens`;

  const webPageJsonLd = {
    "@context": "https://schema.org",
    "@type": "WebPage",
    name: pageTitle,
    description,
    url: canonicalUrl,
    isPartOf: {
      "@type": "WebSite",
      name: "RespiLens",
      url: SITE_URL,
    },
  };

  return (
    <Helmet>
      <title>{pageTitle}</title>
      <meta name="description" content={description} />
      <meta name="robots" content="index,follow,max-image-preview:large" />
      <link rel="canonical" href={canonicalUrl} />

      <meta property="og:type" content={type} />
      <meta property="og:locale" content="en_US" />
      <meta property="og:site_name" content="RespiLens" />
      <meta property="og:title" content={pageTitle} />
      <meta property="og:description" content={description} />
      <meta property="og:url" content={canonicalUrl} />
      <meta property="og:image" content={imageUrl} />
      <meta property="og:image:alt" content={imageAlt} />

      <meta name="twitter:card" content="summary_large_image" />
      <meta name="twitter:title" content={pageTitle} />
      <meta name="twitter:description" content={description} />
      <meta name="twitter:image" content={imageUrl} />
      <meta name="twitter:image:alt" content={imageAlt} />

      <script type="application/ld+json">
        {JSON.stringify(structuredData || webPageJsonLd)}
      </script>
    </Helmet>
  );
};

export default Seo;

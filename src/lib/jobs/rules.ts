// Values quoted on /jobs/about. They mirror config/scoring_config.json in the data repo,
// which is not served (spec §2). When the latest run used a different config version,
// the page says the numbers may be out of date. Update these when you retune.
export const RULES = {
  configVersion: "1.0.4",
  applyFirst: 80,
  worthReviewing: 65,
  stretch: 50,
  dailyCap: 15,
  holdDays: 7,
  unreviewedDays: 14,
} as const;

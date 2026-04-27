/**
 * Centralized configuration exports
 *
 * This file serves as the single entry point for all application configuration.
 * Import config values from './config' rather than from individual config files.
 *
 * @example
 * import { APP_CONFIG, DATASETS, CHART_CONFIG } from '../config';
 */

// Dataset configuration
import {
  DATASETS,
  getAllViewValues,
  MODEL_COLORS,
  getModelColor,
} from "./datasets";

// Application defaults
import { APP_CONFIG } from "./app";

// Forecastle game settings
import { FORECASTLE_CONFIG } from "./forecastle";

// Tournament settings
import {
  TOURNAMENT_CONFIG,
  TOURNAMENT_REGISTRY,
  ENABLED_TOURNAMENTS,
  getTournamentById,
  getTournamentByPath,
  getChallengeById,
  getChallengeByNumber,
  areAllChallengesCompleted,
} from "./tournament";

// Visualization and chart settings
import { CHART_CONFIG } from "./visualization";

// Re-export all configurations
export { DATASETS, getAllViewValues, MODEL_COLORS, getModelColor };
export { APP_CONFIG };
export { FORECASTLE_CONFIG };
export {
  TOURNAMENT_CONFIG,
  TOURNAMENT_REGISTRY,
  ENABLED_TOURNAMENTS,
  getTournamentById,
  getTournamentByPath,
  getChallengeById,
  getChallengeByNumber,
  areAllChallengesCompleted,
};
export { CHART_CONFIG };

/**
 * Convenience function to get the entire configuration object
 * Useful for debugging or when you need access to all configs at once
 *
 * @returns {Object} Object containing all configuration modules
 */
export const getConfig = () => ({
  app: APP_CONFIG,
  forecastle: FORECASTLE_CONFIG,
  chart: CHART_CONFIG,
  datasets: DATASETS,
});

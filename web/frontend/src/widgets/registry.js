// Widget registry: dashboard.layout `type` string -> Svelte component.
// Adding a widget = write the component + one line here + a layout entry
// in web/config.yaml. Unknown types render a visible error card.

import CurrentConditions from './CurrentConditions.svelte';
import AllSkyImage from './AllSkyImage.svelte';
import HistoryChart from './HistoryChart.svelte';
import OpenMeteoForecast from './OpenMeteoForecast.svelte';
import Astronomy from './Astronomy.svelte';
import LoraNodes from './LoraNodes.svelte';
import LoraLinkChart from './LoraLinkChart.svelte';
import ForecastChart from './ForecastChart.svelte';
import RainMap from './RainMap.svelte';
import RawObservations from './RawObservations.svelte';
import SystemStatus from './SystemStatus.svelte';
import CloudClasses from './CloudClasses.svelte';
import Timelapse from './Timelapse.svelte';
import Orrery from './Orrery.svelte';

export const WIDGETS = {
  'current-conditions': CurrentConditions,
  'allsky-image': AllSkyImage,
  'history-chart': HistoryChart,
  'open-meteo-forecast': OpenMeteoForecast,
  'astronomy': Astronomy,
  'lora-nodes': LoraNodes,
  'lora-link-chart': LoraLinkChart,
  'forecast-chart': ForecastChart,
  'rain-map': RainMap,
  'raw-observations': RawObservations,
  'system-status': SystemStatus,
  'cloud-classes': CloudClasses,
  'timelapse': Timelapse,
  'orrery': Orrery,
};

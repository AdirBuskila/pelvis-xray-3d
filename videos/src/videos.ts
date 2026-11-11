// Registry of the series. Each video folder exports its component and its timeline (`tl.total` seconds).
import type { ComponentType } from "react";
import { V00, tl as t00 } from "./v00/Video";
import { V01, tl as t01 } from "./v01/Video";
import { V02, tl as t02 } from "./v02/Video";
import { V03, tl as t03 } from "./v03/Video";
import { V04, tl as t04 } from "./v04/Video";
import { V05, tl as t05 } from "./v05/Video";
import { V06, tl as t06 } from "./v06/Video";
import { V07, tl as t07 } from "./v07/Video";
import { V08, tl as t08 } from "./v08/Video";
import { V09, tl as t09 } from "./v09/Video";
import { V10, tl as t10 } from "./v10/Video";
import { V11, tl as t11 } from "./v11/Video";
import { V12, tl as t12 } from "./v12/Video";
import { V13, tl as t13 } from "./v13/Video";
import { V14, tl as t14 } from "./v14/Video";

export const VIDEOS: { id: string; component: ComponentType; seconds: number; localOnly?: boolean }[] = [
  { id: "V00", component: V00, seconds: t00.total },
  { id: "V01", component: V01, seconds: t01.total },
  { id: "V02", component: V02, seconds: t02.total },
  { id: "V03", component: V03, seconds: t03.total },
  { id: "V04", component: V04, seconds: t04.total },
  { id: "V05", component: V05, seconds: t05.total },
  { id: "V06", component: V06, seconds: t06.total },
  { id: "V07", component: V07, seconds: t07.total },
  { id: "V08", component: V08, seconds: t08.total },
  { id: "V09", component: V09, seconds: t09.total },
  { id: "V10", component: V10, seconds: t10.total },
  { id: "V11", component: V11, seconds: t11.total },
  { id: "V12", component: V12, seconds: t12.total },
  // V13 uses DeepFluoro (CC BY-NC-ND): its assets are never committed, so it renders locally only
  { id: "V13", component: V13, seconds: t13.total, localOnly: true },
  { id: "V14", component: V14, seconds: t14.total },
];

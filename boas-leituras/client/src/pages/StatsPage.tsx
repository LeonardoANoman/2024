import React from "react";
import { StatsCharts } from "../components/StatsCharts";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from "../components/ui/card";

export const StatsPage: React.FC = () => {
  return (
    <div className="container mx-auto p-4">
      <Card>
        <CardHeader>
          <CardTitle>Reading Statistics</CardTitle>
        </CardHeader>
        <CardContent>
          <StatsCharts />
        </CardContent>
      </Card>
    </div>
  );
};
